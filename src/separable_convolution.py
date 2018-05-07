import torch as t

from config import OUTPUT_1D_KERNEL_SIZE as filter_size


class SeparableConvolutionSlow(t.autograd.Function):
    def __init__(self):
        super(SeparableConvolutionSlow, self).__init__()
    
    def forward(self, im, vertical, horizontal):
        n_b = im.size(0)
        n_channels = im.size(1)
        h = im.size(2)
        w = im.size(3)
        h_out = h - filter_size + 1
        w_out = w - filter_size + 1
        
        assert vertical.size(0) == horizontal.size(0) == n_b
        assert vertical.size(1) == horizontal.size(1) == filter_size
        assert horizontal.size(2) == h
        assert horizontal.size(3) == vertical.size(3) == w_out
        assert vertical.size(2) == h_out
        
        output = t.zeros((n_b, n_channels, h_out, w_out))
        
        for b in range(n_b):
            for c in range(n_channels):
                conv_horiz = local_conv_1d_slow(im[b, c], horizontal[b])
                conv_both = local_conv_1d_slow(conv_horiz.transpose(0, 1), vertical[b].transpose(1, 2)).transpose(0, 1)
                output[b, c] = conv_both
        return output


def local_conv_1d_slow(im, filt, output=None):
    h, w = im.shape
    fw, _, _ = filt.shape
    assert filt.shape[1:] == (h, w - fw + 1)
    assert fw % 2 == 1
    fw2 = fw // 2
    if output is None:
        output = t.zeros((h, w - fw + 1))
    for i in range(h):
        for j in range(fw2, w - fw2):
            output[i, j - fw2] = t.dot(im[i, j - fw2:j + fw2 + 1], filt[:, i, j - fw2])
    return output
