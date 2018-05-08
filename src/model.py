#
# KTH Royal Institute of Technology
#

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules.loss import _Loss, _assert_no_grad
from src.separable_convolution import SeparableConvolutionSlow
from libs.sepconv import SeparableConvolution
import src.config as config

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # self.relu = nn.ReLU()
        # self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        # self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        if torch.cuda.is_available():
            self.separable_conv = SeparableConvolution()
        else:
            self.separable_conv = SeparableConvolutionSlow()

        self._initialize_weights()

    def forward(self, x):
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.relu(self.conv3(x))
        # x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        print('_initialize_weights')
        # init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        # init.orthogonal_(self.conv4.weight)

class CustomLoss(_Loss):

    def __init__(self, size_average=True):
        super(CustomLoss, self).__init__(size_average)

    def forward(self, input, target):
        _assert_no_grad(target)
        # ...
        # Return the loss as a Tensor
        return None
