import torch.nn.functional as F
import torch.nn as nn
import torch


def mse_loss(output, target):
    return F.mse_loss(output, target)

# class RMSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()
#
#     def forward(self, yhat, y):
#         return torch.sqrt(self.mse(yhat, y))
