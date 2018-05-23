import torch as t
import os
import torch.multiprocessing as mp
from src.config import NUM_WORKERS, OUTPUT_1D_KERNEL_SIZE as FILTER_SIZE


class SeparableConvolutionSlow(t.nn.Module):
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

        n_workers = NUM_WORKERS

        if os.name == 'nt' and NUM_WORKERS > 1:
            print('Parallel Separable Convolution on CPU not supported on Windows. Proceeding on main thread...')
            n_workers = 1

        if vertical.requires_grad and NUM_WORKERS > 1:
            print('Parallel Separable Convolution on CPU not supported during training. Proceeding on main thread...')
            n_workers = 1

        output = im.new().resize_(n_b, n_channels, m_out, m_out).zero_()

        if n_workers > 1:
            return parallel_sep_conv(im, horizontal, vertical, output, n_workers)
        else:
            return sep_conv(im, horizontal, vertical, output)


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
            local_vert = vertical[:, row, col].view(-1, 1)
            output[:, row, col] = (sub_patch * local_horiz * local_vert).sum(dim=1).sum(dim=1)
    return output


def _sep_conv_worker(im, horizontal, vertical, output, worker_batch_size, offset):
    n_b = im.size(0)
    max_range = min(n_b, worker_batch_size+offset)
    for b in range(offset, max_range):
        local_separable_conv_2d(im[b], horizontal[b], vertical[b], output=output[b])
    return output


def sep_conv(im, horizontal, vertical, output):
    """
    Runs the separable convolution on multiple images sequentially on a single thread
    :param im: Input images as a tensor. im[0] must correspond to the first image of the batch
    :param horizontal: Set of horizontal filters as a tensor
    :param vertical: Set of vertical filters as a tensor
    :param output: Tensor used as output. Same shape as im. Must be passed pre-allocated and initialized with zeros
    :return: Tensor resulting from the convolution
    """
    return _sep_conv_worker(im, horizontal, vertical, output, im.size(0), 0)


def parallel_sep_conv(im, horizontal, vertical, output, n_workers):
    """
    Spawns the specified amount of workers to run the separable convolution on multiple images in parallel
    :param im: Input images as a tensor. im[0] must correspond to the first image of the batch
    :param horizontal: Set of horizontal filters as a tensor
    :param vertical: Set of vertical filters as a tensor
    :param output: Tensor used as output. Same shape as im. Must be passed pre-allocated and initialized with zeros
    :param n_workers: Number of workers to be used. Must be greater than zero
    :return: Tensor resulting from the convolution
    """

    n_b = im.size(0)
    n_workers = min(n_b, n_workers)
    worker_batch_size = n_b // n_workers
    processes = []

    output.share_memory_()

    for i in range(n_workers):

        offset = worker_batch_size * i
        if i == n_workers-1:
            worker_batch_size += n_b % n_workers

        p = mp.Process(target=_sep_conv_worker, args=(im, horizontal, vertical, output, worker_batch_size, offset,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return output
