#
# KTH Royal Institute of Technology
#

import argparse
import torch
import math
from torchvision.transforms import CenterCrop
from os.path import join, isdir
from timeit import default_timer as timer
from src.utilities import write_video
from src.interpolate import interpolate_batch
from src.data_manager import load_tuples
from src.extract_frames import extract_frames


def interpolate_video(src_path, dest_path, model_path, input_fps=None, input_limit=None, batch_size=None):

    from src.model import Net

    tick_t = timer()

    print('===> Loading model...')
    model = Net.from_file(model_path)

    if isdir(src_path):
        if input_fps is None:
            raise Exception('Argument --inputfps is required if the source is a folder of frames')
        print('===> Reading frames...')
        input_frames = load_tuples(src_path, 1, 1, paths_only=False)
        input_frames = [x[0] for x in input_frames]
    else:
        print('===> Reading video...')
        input_frames, detected_fps = extract_frames(src_path)
        if detected_fps is None:
            if input_fps is None:
                raise Exception('Argument --inputfps is required for this type of source')
        else:
            input_fps = detected_fps


    if input_limit is not None:
        input_frames = input_frames[:input_limit]
    n_input_frames = len(input_frames)

    if not torch.cuda.is_available():
        crop_size = min(input_frames[0].size)
        crop = CenterCrop(crop_size)
        print(f'===> CUDA not available. Cropping input as {crop_size}x{crop_size}...')
        input_frames = [crop(x) for x in input_frames]

    if batch_size is not None and batch_size > 1:
        batch_size = min(batch_size, n_input_frames)
    else:
        batch_size = n_input_frames

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
        frame2 = input_frames[i+1]
        middle = middle_frames[i]
        output_frames += [middle, frame2]
        print('Frame {}/{} done'.format(i+1, iters))

    if isdir(dest_path):
        print('===> Saving frames...')
        for i, frame in enumerate(output_frames):
            file_name = '{:07d}.jpg'.format(i)
            file_path = join(dest_path, file_name)
            frame.save(file_path)
    else:
        print('===> Saving video...')
        write_video(dest_path, output_frames, fps=(input_fps * 2))

    tock_t = timer()
    print("Done. Took ~{}s".format(round(tock_t - tick_t)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Video Frame Interpolation')
    parser.add_argument('--src', dest='src_path', type=str, required=True, help='path to the video, either as a single file or as a folder')
    parser.add_argument('--dest', dest='dest_path', type=str, required=True, help='output path of the resulting video, either as a single file or as a folder')
    parser.add_argument('--model', dest='model_path', type=str, required=True, help='path of the trained model')
    parser.add_argument('--inputfps', dest='input_fps', type=int, required=False, default=None, help='frame-rate of the input. Only used if the frames are read from a folder')
    parser.add_argument('--inputlimit', dest='input_limit', type=int, required=False, default=None, help='maximum number of processed input frames')
    parser.add_argument('--batchsize', dest='batch_size', type=int, required=False, default=None, help='number of frames to be processed at the same time (i.e. number of interpolations in parallel +1)')
    interpolate_video(**vars(parser.parse_args()))
