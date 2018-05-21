#
# KTH Royal Institute of Technology
#

import torch


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a - b) ** 2).mean()


def psnr(approx: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the Peak Signal-to-Noise ratio between two images
    :param approx: Approximated image as a tensor
    :param target: Target image as a tensor
    :return: PSNR as a tensor
    """
    _mse = mse(approx, target)
    _max = target.max()
    return 20 * _max.log10() - 10 * _mse.log10()
