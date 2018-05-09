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
        conv_kernel_size = (3, 3)
        conv_stride = (1, 1)
        conv_padding = 1

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv32 = nn.Conv2d(6, 32, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.conv64 = nn.Conv2d(32, 64, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.conv128 = nn.Conv2d(64, 128, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.conv256 = nn.Conv2d(128, 256, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.conv512 = nn.Conv2d(256, 512, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.conv512x512 = nn.Conv2d(512, 512, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.upconv256 = nn.Conv2d(512, 256, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.upconv128 = nn.Conv2d(256, 128, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.upconv64 = nn.Conv2d(128, 64, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)

        self.upconv51_1 = nn.Conv2d(64, 51, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.upconv51_2 = nn.Conv2d(64, 51, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.upconv51_3 = nn.Conv2d(64, 51, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
        self.upconv51_4 = nn.Conv2d(64, 51, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)

        if torch.cuda.is_available():
            self.separable_conv = SeparableConvolution()
        else:
            self.separable_conv = SeparableConvolutionSlow()

        self._initialize_weights()

    def forward(self, x):
        i1 = x[:, :3]
        i2 = x[:, 3:6]

        print('_start_pass')
        x = self.relu(self.conv32(x))
        x = self.pool(x)

        x64 = self.relu(self.conv64(x))
        x128 = self.pool(x64)
        print('_conv_64')

        x128 = self.relu(self.conv128(x128))
        x256 = self.pool(x128)
        print('_conv_128')

        x256 = self.relu(self.conv256(x256))
        x512 = self.pool(x256)

        print('_conv_256')

        x512 = self.relu(self.conv512(x512))
        x = self.pool(x512)

        print('_conv_512')

        x = self.relu(self.conv512x512(x))
        x = self.pool(x)

        print('_conv_512x512')

        x = self.upsamp(x)
        x += x512
        x = self.relu(self.upconv256(x))

        print('_up_conv256')

        x = self.upsamp(x)
        x += x256
        x = self.relu(self.upconv128(x))

        print('_up_conv128')
        x = self.upsamp(x)
        x += x128
        x = self.relu(self.upconv64(x))
        x += x64
        x = self.upsamp(x)

        print('_up_conv64')
        k2h = self.relu(self.upconv51_1(x))
        k2h = self.upsamp(k2h)

        print('_up_conv_51_1')

        k2v = self.relu(self.upconv51_2(x))
        k2v = self.upsamp(k2v)

        print('_up_conv_51_2')

        k1h = self.relu(self.upconv51_3(x))
        k1h = self.upsamp(k1h)

        print('_up_conv_51_3')

        k1v = self.relu(self.upconv51_4(x))
        k1v = self.upsamp(k1v)

        print('_up_conv_51_4')
        return self.separable_conv(i2, k2v, k2h) + self.separable_conv(i1, k1v, k1h)

    def _initialize_weights(self):
        print('_initialize_weights')
        gain = init.calculate_gain('relu')
        init.orthogonal_(self.conv32.weight, gain)
        init.orthogonal_(self.conv64.weight, gain)
        init.orthogonal_(self.conv128.weight, gain)
        init.orthogonal_(self.conv256.weight, gain)
        init.orthogonal_(self.conv512.weight, gain)
        init.orthogonal_(self.conv512x512.weight, gain)
        init.orthogonal_(self.upconv64.weight, gain)
        init.orthogonal_(self.upconv128.weight, gain)
        init.orthogonal_(self.upconv256.weight, gain)
        init.orthogonal_(self.upconv51_1.weight, gain)
        init.orthogonal_(self.upconv51_2.weight, gain)
        init.orthogonal_(self.upconv51_3.weight, gain)
        init.orthogonal_(self.upconv51_4.weight, gain)


class CustomLoss(_Loss):

    def __init__(self, size_average=True):
        super(CustomLoss, self).__init__(size_average)

    def forward(self, input, target):
        _assert_no_grad(target)
        # ...
        # Return the loss as a Tensor
        return None
