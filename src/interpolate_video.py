#
# KTH Royal Institute of Technology
#

import argparse
import torch
import math
import numpy as np
import cv2 as cv
from torchvision.transforms import CenterCrop
from os.path import join, isdir
from timeit import default_timer as timer
from src.data_manager import load_tuples
from src.interpolate import interpolate_batch
from src.extract_frames import extract_frames


def _write_video(file_path, frames, fps):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    pil_to_numpy = lambda x: np.array(x)[:, :, ::-1]

    w, h = frames[0].size
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(pil_to_numpy(frame))

    writer.release()


if __name__ == '__main__':

    from src.model import Net

    parser = argparse.ArgumentParser(description='Video Frame Interpolation')
    parser.add_argument('--src', type=str, required=True, help='path to the video, either as a single file or as a folder')
    parser.add_argument('--dest', type=str, required=True, help='output path of the resulting video, either as a single file or as a folder')
    parser.add_argument('--model', type=str, required=True, help='path of the trained model')
    parser.add_argument('--inputfps', type=int, required=False, default=None, help='frame-rate of the input. Only used if the frames are read from a folder')
    parser.add_argument('--inputlimit', type=int, required=False, default=None, help='maximum number of processed input frames')
    parser.add_argument('--batchsize', type=int, required=False, default=None, help='number of frames to be processed at the same time (i.e. number of interpolations in parallel +1)')
    params = parser.parse_args()

    tick_t = timer()

    print('===> Loading model...')
    model = Net()
    state_dict = torch.load(params.model)
    model.load_state_dict(state_dict)

    if isdir(params.src):
        if params.inputfps is None:
            raise Exception('Argument --inputfps is required if the source is a folder of frames')
        print('===> Reading frames...')
        input_fps = params.inputfps
        input_frames = load_tuples(params.src, 1, 1, paths_only=False)
        input_frames = [x[0] for x in input_frames]
    else:
        print('===> Reading video...')
        input_frames, input_fps = extract_frames(params.src)

    if params.inputlimit is not None:
        input_frames = input_frames[:params.inputlimit]
    n_input_frames = len(input_frames)

    if not torch.cuda.is_available():
        crop_size = min(input_frames[0].size)
        crop = CenterCrop(crop_size)
        print(f'===> CUDA not available. Cropping input as {crop_size}x{crop_size}...')
        input_frames = [crop(x) for x in input_frames]

    batch_size = n_input_frames
    if params.batchsize is not None and params.batchsize > 1:
        batch_size = min(params.batchsize, batch_size)

    # FIXME: Change this monstrosity to something more elegant
    n_batches = int(math.ceil(1.0 * n_input_frames / (batch_size - 1)))
    if (batch_size-1)*(n_batches-1) >= n_input_frames - 1:
        n_batches -= 1
    print(f'Job split into {n_batches} batches')

    print('===> Interpolating...')
    middle_frames = []
    for i in range(n_batches):
        idx = (batch_size-1)*i
        batch = input_frames[idx : idx+batch_size]
        middle_frames += interpolate_batch(model, batch)
        print('Batch {}/{} done'.format(i+1, n_batches))

    print('===> Stitching frames...')
    output_frames = input_frames[:1]
    iters = len(middle_frames)
    for i in range(iters):
        frame1 = input_frames[i]
        frame2 = input_frames[i+1]
        middle = middle_frames[i]
        output_frames += [middle, frame2]
        print('Frame {}/{} done'.format(i+1, iters))

    if isdir(params.dest):
        print('===> Saving frames...')
        for i, frame in enumerate(output_frames):
            file_name = '{:07d}.jpg'.format(i)
            file_path = join(params.dest, file_name)
            frame.save(file_path)
    else:
        print('===> Saving video...')
        _write_video(params.dest, output_frames, fps=(input_fps * 2))

    tock_t = timer()

    print("Done. Took ~{}s".format(round(tock_t - tick_t)))
