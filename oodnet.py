import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np


class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


class OODNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 5, stride=2, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        """
        self.conv4 = nn.Conv2d(128, 256, 5, bias=False, padding=2)
        self.bn2d4 = nn.BatchNorm2d(256, eps=1e-04, affine=False)
        
        self.conv5 = nn.Conv2d(256, 512, 5, bias=False, padding=2)
        self.bn2d5 = nn.BatchNorm2d(512, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(512, 1024, 5, bias=False, padding=2)
        self.bn2d6 = nn.BatchNorm2d(1024, eps=1e-04, affine=False)
        """
        self.fc1 = nn.Linear(128 * 8 * 8, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        """
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))
        x = self.conv6(x)
        x = self.pool(F.leaky_relu(self.bn2d6(x)))                
        """
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x