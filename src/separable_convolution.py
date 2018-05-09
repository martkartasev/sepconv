import torch as t

from src.config import OUTPUT_1D_KERNEL_SIZE as FILTER_SIZE


class SeparableConvolutionSlow(t.autograd.Function):
    def __init__(self):
        super(SeparableConvolutionSlow, self).__init__()
    
    def forward(self, im, vertical, horizontal):
        n_b = im.size(0)
        n_channels = im.size(1)
        m = im.size(2)
        m_out = m - FILTER_SIZE + 1
        
        assert im.size(2) == im.size(3)
        assert vertical.size(0) == horizontal.size(0) == n_b
        assert vertical.size(1) == horizontal.size(1) == FILTER_SIZE
        assert vertical.size(2) == horizontal.size(2) == vertical.size(3) == horizontal.size(3) == m_out
        
        output = input.new().resize_(n_b, n_channels, m_out, m_out).zeros_()
        
        for b in range(n_b):
             local_separable_conv_2d(im[b], horizontal[b], vertical[b], output=output[b])
        return output
    
    def backward(self, grad_output):
        raise NotImplementedError


def local_separable_conv_2d(im, horizontal, vertical, output=None):
    """im: [n_channels x m x m], horizontal: [51 x m x m], vertical: [51 x m x m]
       -> return: [n_channels x (m - 50) x (m - 50)]"""
    n_channels = im.size(0)
    m = im.size(1)
    m_out = m - FILTER_SIZE + 1
    if output is None:
        output = t.zeros((n_channels, m_out, m_out))
    for row in range(m_out):
        for col in range(m_out):
            sub_patch = im[:, row:row + FILTER_SIZE, col:col + FILTER_SIZE]
            local_horiz = horizontal[:, row, col]
            local_vert = vertical[:, row, col].reshape(-1, 1)
            output[:, row, col] = (sub_patch * local_horiz * local_vert).sum(dim=1).sum(dim=1)
    return output