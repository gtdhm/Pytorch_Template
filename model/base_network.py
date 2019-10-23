# ==================model.base_network.py=====================
# This module implements a base network for the project.

# Version: 1.0.0
# Date: 2019.05.20
# ============================================================

import torch.nn as nn


###############################################################
# BaseNetwork Class
###############################################################
class BaseNetwork(nn.Module):
    """This class includes base architecture for the network.
    Inputs:
        cfg: the total options
    Examples:
        <<< network = BaseNetwork()
            out = network(input)
    """

    def __init__(self, num_classes):
        super(BaseNetwork, self).__init__()

        # TODO(Usr) >>> redefine following layers
        # Define network layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=16,
                      kernel_size=(4, 3),
                      stride=1,
                      padding=0),           # (16, 220, 220)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     # (16, 110, 110)

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(4, 4),
                      stride=1,
                      padding=0),           # (32, 106, 106)
            nn.ReLU(),
            nn.MaxPool2d(2)                 # (32, 53, 53)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=2,
                      stride=1,
                      padding=0),           # (64, 51, 51)
            nn.ReLU(),
            nn.MaxPool2d(3)                 # (64, 17, 17)
        )
        self.out = nn.Sequential(
            nn.Linear(64 * 17 * 17, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        # TODO(User): End

    def forward(self, x):
        # TODO(Usr) >>> redefine following forward step
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
        # TODO(User): End



