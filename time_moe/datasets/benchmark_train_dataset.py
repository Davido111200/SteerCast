#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os

import numpy as np
import pandas as pd

from .ts_dataset import TimeSeriesDataset


class BenchmarkTrainDataset(TimeSeriesDataset):
    """Fine-tuning data for a benchmark CSV (first column = timestamp).

    Every channel is one training sequence covering all rows before the test
    split (training + validation rows: ETT uses the first 16 months, every other
    dataset the first 80% of the rows). Rows containing NaNs are dropped first.
    Sequences are left unscaled; ``TimeMoEDataset`` normalises each one.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

        df = pd.read_csv(csv_path)
        df = df[df.columns[1:]].dropna(axis=0, how="any")
        end = _non_test_end(csv_path, len(df))

        self.data = [df[col].values[:end].astype(np.float64) for col in df.columns]
        self.num_sequences = len(self.data)
        self.num_tokens = int(sum(len(seq) for seq in self.data))

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        return self.data[seq_idx]

    def get_num_tokens(self):
        return self.num_tokens

    def get_sequence_length_by_idx(self, seq_idx):
        return len(self.data[seq_idx])

    @staticmethod
    def is_valid_path(data_path):
        return os.path.isfile(data_path) and data_path.lower().endswith(".csv")


def _non_test_end(csv_path: str, total_len: int) -> int:
    base_name = os.path.basename(csv_path).lower()
    if "etth" in base_name:
        return 12 * 30 * 24 + 4 * 30 * 24
    if "ettm" in base_name:
        return 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4
    return total_len - int(total_len * 0.2)
