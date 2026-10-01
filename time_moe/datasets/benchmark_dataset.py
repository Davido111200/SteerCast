#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""Benchmark datasets used by SteerCast and the baselines.

All CSV benchmarks follow the standard long-term forecasting splits
(ETT: 12/4/4 months; everything else: 70/10/20 chronological split), are
standardised with statistics of the training split, and are processed
channel-independently: every channel of every window is one univariate sample.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from time_moe.datasets.general_dataset import GeneralDataset


def compute_split_borders(csv_path, seq_len: int):
    """Return ``(border1s, border2s, n_rows)`` for the (train, valid, test) splits.

    ``border*s`` are absolute row indices into the CSV. The valid and test splits
    start ``seq_len`` rows early so that their first window has a full look-back.
    """
    df = pd.read_csv(csv_path)
    n_rows = len(df)
    base_name = os.path.basename(csv_path).lower()
    if 'etth' in base_name:
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif 'ettm' in base_name:
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        num_train = int(n_rows * 0.7)
        num_test = int(n_rows * 0.2)
        num_vali = n_rows - num_train - num_test
        border1s = [0, num_train - seq_len, n_rows - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, n_rows]
    return border1s, border2s, n_rows


SPLITS = {'train': 0, 'val': 1, 'test': 2}


class BenchmarkEvalDataset(Dataset):
    """Sliding windows over the (standardised) test or validation split, one channel at a time."""

    def __init__(self, csv_path, seq_len: int, pred_len: int, split: str = 'test'):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        df = pd.read_csv(csv_path)
        border1s, border2s, _ = compute_split_borders(csv_path, seq_len)

        if 'm4' in csv_path.lower():
            df["series_value"] = df["series_value"].apply(json.loads)
            df_values = df['series_value']
        else:
            cols = df.columns[1:]
            df_values = df[cols].values

        split_id = SPLITS[split]
        train_data = df_values[border1s[0]:border2s[0]]
        eval_data = df_values[border1s[split_id]:border2s[split_id]]

        scaler = StandardScaler()
        scaler.fit(train_data)
        scaled_eval_data = scaler.transform(eval_data)

        self.hf_dataset = scaled_eval_data.transpose(1, 0)  # [C, T_split]
        self.num_sequences = len(self.hf_dataset)
        self.window_length = self.seq_len + self.pred_len

        self.sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.hf_dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.hf_dataset[seq_i]
        window_seq = np.array(seq[offset_i - self.window_length: offset_i], dtype=np.float32)
        return {
            'inputs': np.array(window_seq[: self.seq_len], dtype=np.float32),
            'labels': np.array(window_seq[-self.pred_len:], dtype=np.float32),
            'channel_id': np.array(seq_i),
        }


class BenchmarkEvalDatasetTrain(Dataset):
    """Sliding windows over the (standardised) training split.

    This is the retrieval corpus of SteerCast and of the retrieval baselines.

    Two access paths are provided and enumerate windows in the same order
    (channel-major, then time):
      * ``__getitem__`` returns ``{'inputs', 'labels', 'channel_id'}`` dicts;
      * ``self.data_loader`` yields ``(seq_x, seq_y, seq_x_mark, seq_y_mark)``
        tuples in the layout of Time-Series-Library's ``Dataset_ETT_hour``.

    Args:
        args: namespace providing ``batch_size`` and (optionally) ``label_len``.
        max_train_samples: keep only the last ``max_train_samples`` time steps of
            the training split (``None`` keeps everything).
        temporal_buffer: drop the last ``temporal_buffer`` windows of every
            channel, enlarging the gap between the database and the test split
            (used for the strict temporal-isolation ablation).
    """

    def __init__(self, args, csv_path, seq_len: int, pred_len: int, max_train_samples=None,
                 temporal_buffer: int = 0):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.args = args
        self.label_len = getattr(args, 'label_len', 0)
        self.temporal_buffer = int(temporal_buffer)

        df = pd.read_csv(csv_path)
        border1s, border2s, _ = compute_split_borders(csv_path, seq_len)

        df_stamp = df[['date']].iloc[border1s[0]:border2s[0]].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        self.data_stamp = pd.DataFrame({
            'month': df_stamp['date'].dt.month,
            'day': df_stamp['date'].dt.day,
            'weekday': df_stamp['date'].dt.weekday,
            'hour': df_stamp['date'].dt.hour,
        }).values.astype(np.float32)  # [T_train, 4]

        cols = df.columns[1:]
        df_values = df[cols].values

        train_data = df_values[border1s[0]:border2s[0]]
        if max_train_samples is not None:
            train_data = train_data[-max_train_samples:, :]

        scaler = StandardScaler()
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)

        self.hf_dataset = scaled_train_data.transpose(1, 0)  # [C, T_train]
        self.num_sequences = len(self.hf_dataset)
        self.window_length = self.seq_len + self.pred_len

        self.sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.hf_dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            max_offset = n_points - self.temporal_buffer
            for offset_idx in range(self.window_length, max_offset):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

        if torch.cuda.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset=self.sub_seq_indexes, shuffle=False)
        else:
            sampler = None

        self.data_loader = DataLoader(
            self.sub_seq_indexes,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=2,
            sampler=sampler,
            prefetch_factor=2,
            drop_last=False,
            collate_fn=self._collate_like_ett_hour,
        )

    def _collate_like_ett_hour(self, batch_idx_tuples):
        """Collate ``(seq_idx, offset_idx)`` pairs into ``Dataset_ETT_hour`` tuples.

        Returns:
            seq_x:      [B, seq_len, 1]
            seq_y:      [B, label_len + pred_len, 1]
            seq_x_mark: [B, seq_len, 4]               (month, day, weekday, hour)
            seq_y_mark: [B, label_len + pred_len, 4]
        """
        B = len(batch_idx_tuples)
        Lx = self.seq_len
        Ld = self.label_len
        Ly = self.pred_len
        Dm = self.data_stamp.shape[1]

        seq_x = np.empty((B, Lx, 1), dtype=np.float32)
        seq_y = np.empty((B, Ld + Ly, 1), dtype=np.float32)
        seq_x_mark = np.empty((B, Lx, Dm), dtype=np.float32)
        seq_y_mark = np.empty((B, Ld + Ly, Dm), dtype=np.float32)

        window_len = Lx + Ly
        for i, (seq_i, offset_i) in enumerate(batch_idx_tuples):
            # The window is [offset - (seq_len + pred_len), offset) on the train axis.
            s_begin = offset_i - window_len
            s_end = s_begin + Lx
            r_begin = s_end - Ld
            r_end = r_begin + Ld + Ly

            window = np.asarray(self.hf_dataset[seq_i][s_begin:offset_i], dtype=np.float32)
            seq_x[i, :, 0] = window[:Lx]
            seq_y[i, :, 0] = window[Lx - Ld:]
            seq_x_mark[i] = self.data_stamp[s_begin:s_end]
            seq_y_mark[i] = self.data_stamp[r_begin:r_end]

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(seq_x_mark),
            torch.from_numpy(seq_y_mark),
        )

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.hf_dataset[seq_i]
        window_seq = np.array(seq[offset_i - self.window_length: offset_i], dtype=np.float32)
        return {
            'inputs': np.array(window_seq[: self.seq_len], dtype=np.float32),
            'labels': np.array(window_seq[-self.pred_len:], dtype=np.float32),
            'channel_id': np.array(seq_i),
        }


class GeneralEvalDataset(Dataset):
    """Sliding windows over a non-CSV dataset (jsonl / npy / pkl), see ``GeneralDataset``."""

    def __init__(self, data_path, seq_len: int, pred_len: int, onfly_norm: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.onfly_norm = onfly_norm
        self.window_length = self.seq_len + self.pred_len
        self.dataset = GeneralDataset(data_path)

        self.sub_seq_indexes = []
        for seq_idx, seq in enumerate(self.dataset):
            n_points = len(seq)
            if n_points < self.window_length:
                continue
            for offset_idx in range(self.window_length, n_points):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        seq_i, offset_i = self.sub_seq_indexes[idx]
        seq = self.dataset[seq_i]
        window_seq = np.array(seq[offset_i - self.window_length: offset_i], dtype=np.float32)
        return {
            'inputs': np.array(window_seq[: self.seq_len], dtype=np.float32),
            'labels': np.array(window_seq[-self.pred_len:], dtype=np.float32),
            'channel_id': np.array(seq_i),
        }
