"""Running MSE / MAE accumulators (sums over all forecast points)."""
import torch


class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels):
        self.value += self._calculate(preds, labels)

    def _calculate(self, preds, labels):
        raise NotImplementedError


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels):
        return torch.sum(torch.abs(preds - labels))
