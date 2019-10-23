# ==================model.base_loss.py=====================
# This module implements some defined losses which are not
# in the Pytorch.

# Version: 1.0.0
# Date: 2019.05.20
# ============================================================

import torch
import torch.nn as nn


###############################################################
# Focal Loss
###############################################################
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, y_input, target):
        logp = self.ce(y_input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
