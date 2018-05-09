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

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv32 = nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(1, 1))
        self.conv64 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.conv128 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
        self.conv256 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
        self.conv512 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1))
        self.conv512x512 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))
        self.upconv256 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1))
        self.upconv128 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1))
        self.upconv64 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1))

        self.upconv51_1 = nn.Conv2d(64, 51, kernel_size=(3, 3), stride=(1, 1))
        self.upconv51_2 = nn.Conv2d(64, 51, kernel_size=(3, 3), stride=(1, 1))
        self.upconv51_3 = nn.Conv2d(64, 51, kernel_size=(3, 3), stride=(1, 1))
        self.upconv51_4 = nn.Conv2d(64, 51, kernel_size=(3, 3), stride=(1, 1))

        if torch.cuda.is_available():
            self.separable_conv = SeparableConvolution()
        else:
            self.separable_conv = SeparableConvolutionSlow()

        self._initialize_weights()

    def forward(self, x):
        i1 = x[:, :3]
        i2 = x[:, 3:6]

        x = self.relu(self.conv32(x))
        x = self.pool(x)
        x = self.relu(self.conv64(x))
        x = self.pool(x)

        x = self.relu(self.conv128(x))
        x = self.pool(x)

        x = self.relu(self.conv256(x))
        x = self.pool(x)

        x = self.relu(self.conv512(x))
        x = self.pool(x)

        x = self.relu(self.conv512x512(x))
        x = self.pool(x)

        x = self.upsamp(x)
        x = self.relu(self.upconv256(x))

        x = self.upsamp(x)
        x = self.relu(self.upconv128(x))

        x = self.upsamp(x)
        x = self.relu(self.upconv64(x))
        x = self.upsamp(x)

        k2h = self.relu(self.upconv51_1(x))
        k2h = self.upsamp(k2h)

        k2v = self.relu(self.upconv51_2(x))
        k2v = self.upsamp(k2v)

        k1h = self.relu(self.upconv51_3(x))
        k1h = self.upsamp(k1h)

        k1v = self.relu(self.upconv51_4(x))
        k1v = self.upsamp(k1v)

        return self.separable_conv(i2, k2v, k2h) + self.separable_conv(i1, k1v, k1h)

    def _initialize_weights(self):
        print('_initialize_weights')
        init.orthogonal_(self.conv32.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv64.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv128.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv256.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv512.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv512x512.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.upconv64.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.upconv128.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.upconv256.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.upconv51_1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.upconv51_2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.upconv51_3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.upconv51_4.weight, init.calculate_gain('relu'))

class CustomLoss(_Loss):

    def __init__(self, size_average=True):
        super(CustomLoss, self).__init__(size_average)

    def forward(self, input, target):
        _assert_no_grad(target)
        # ...
        # Return the loss as a Tensor
        return None
