#
# KTH Royal Institute of Technology
#

import torch
import numpy as np
import cv2 as cv


def pil_to_cv(pil_image):
    """
    Returns a copy of an image in a representation suited for OpenCV
    :param pil_image: PIL.Image object
    :return: Numpy array compatible with OpenCV
    """
    return np.array(pil_image)[:, :, ::-1]


def write_video(file_path, frames, fps):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    w, h = frames[0].size
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(pil_to_cv(frame))

    writer.release()


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
