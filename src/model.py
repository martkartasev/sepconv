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

        conv_kernel = (3, 3)
        conv_stride = (1, 1)
        conv_padding = 1
        sep_kernel = config.OUTPUT_1D_KERNEL_SIZE

        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv32 = self.conv_module(6, 32, conv_kernel, conv_stride, conv_padding)
        self.conv64 = self.conv_module(32, 64, conv_kernel, conv_stride, conv_padding)
        self.conv128 = self.conv_module(64, 128, conv_kernel, conv_stride, conv_padding)
        self.conv256 = self.conv_module(128, 256, conv_kernel, conv_stride, conv_padding)
        self.conv512 = self.conv_module(256, 512, conv_kernel, conv_stride, conv_padding)
        self.conv512x512 = self.conv_module(512, 512, conv_kernel, conv_stride, conv_padding)
        self.upsamp512 = self.upsample(512, 512, conv_kernel, conv_stride, conv_padding, self.upsamp)
        self.upconv256 = self.conv_module(512, 256, conv_kernel, conv_stride, conv_padding)
        self.upsamp256 = self.upsample(256, 256, conv_kernel, conv_stride, conv_padding, self.upsamp)
        self.upconv128 = self.conv_module(256, 128, conv_kernel, conv_stride, conv_padding)
        self.upsamp128 = self.upsample(128, 128, conv_kernel, conv_stride, conv_padding, self.upsamp)
        self.upconv64 = self.conv_module(128, 64, conv_kernel, conv_stride, conv_padding)
        self.upsamp64 = self.upsample(64, 64, conv_kernel, conv_stride, conv_padding, self.upsamp)
        self.upconv51_1 = self.kernel_conv(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp)
        self.upconv51_2 = self.kernel_conv(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp)
        self.upconv51_3 = self.kernel_conv(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp)
        self.upconv51_4 = self.kernel_conv(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp)

        # FIXME: Use proper padding
        self.pad = nn.ConstantPad2d(sep_kernel // 2, 0.0)

        if torch.cuda.is_available():
            self.separable_conv = SeparableConvolution()
        else:
            self.separable_conv = SeparableConvolutionSlow()

        self._initialize_weights()

    def forward(self, x):
        i1 = x[:, :3]
        i2 = x[:, 3:6]

        print('_start_pass')
        x = self.conv32(x)
        x = self.pool(x)

        x64 = self.conv64(x)
        x128 = self.pool(x64)
        print('_conv_64')

        x128 = self.conv128(x128)
        x256 = self.pool(x128)
        print('_conv_128')

        x256 = self.conv256(x256)
        x512 = self.pool(x256)
        print('_conv_256')

        x512 = self.conv512(x512)
        x = self.pool(x512)
        print('_conv_512')

        x = self.conv512x512(x)
        print('_conv_512x512')

        # -----------------------------------------------------------------------

        x = self.upsamp512(x)
        x += x512
        x = self.upconv256(x)

        print('_up_conv256')

        x = self.upsamp256(x)
        x += x256
        x = self.upconv128(x)

        print('_up_conv128')
        x = self.upsamp128(x)
        x += x128
        x = self.upconv64(x)

        x = self.upsamp64(x)
        x += x64
        print('_up_conv64')

        # --------------------------------

        k2h = self.upconv51_1(x)
        k2h = self.upsamp(k2h)
        print('_up_conv_51_1')

        k2v = self.upconv51_2(x)
        k2v = self.upsamp(k2v)
        print('_up_conv_51_2')

        k1h = self.upconv51_3(x)
        k1h = self.upsamp(k1h)
        print('_up_conv_51_3')

        k1v = self.upconv51_4(x)
        k1v = self.upsamp(k1v)
        print('_up_conv_51_4')

        padded_i2 = self.pad(i2)
        padded_i1 = self.pad(i1)

        print('_up_conv_51_4')
        return self.separable_conv(padded_i2, k2v, k2h) + self.separable_conv(padded_i1, k1v, k1h)

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

    def conv_module(self, in_channels, out_channels, kernel, stride, padding):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding), torch.nn.ReLU(),
        )

    def kernel_conv(self, in_channels, out_channels, kernel, stride, padding, upsample):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding), torch.nn.ReLU(),
            upsample,
            torch.nn.Conv2d(out_channels, out_channels, kernel, stride, padding)
        )

    def upsample(self, in_channels, out_channels, kernel, stride, padding, upsample):
        return torch.nn.Sequential(
            upsample, torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding), torch.nn.ReLU(),
        )


class CustomLoss(_Loss):

    def __init__(self, size_average=True):
        super(CustomLoss, self).__init__(size_average)

    def forward(self, input, target):
        _assert_no_grad(target)
        # ...
        # Return the loss as a Tensor
        return None
