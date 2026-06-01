import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossFunction


class ListNetLossFunction(LossFunction):

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        # Original ListNet: expects (batch_size, list_length) shaped inputs
        # softmax over dim=1 (list dimension)
        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + 1e-10
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))
