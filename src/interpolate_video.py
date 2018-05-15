#
# KTH Royal Institute of Technology
#

import imageio
import argparse
import torch
import math
import numpy as np
from timeit import default_timer as timer
from PIL import Image
from src.interpolate import interpolate_batch


if __name__ == '__main__':

    from src.model import Net

    parser = argparse.ArgumentParser(description='Video Frame Interpolation')
    parser.add_argument('--src', type=str, required=True, help='path to the video')
    parser.add_argument('--dest', type=str, required=True, help='output path of the resulting video')
    parser.add_argument('--model', type=str, required=True, help='path of the trained model')
    parser.add_argument('--maxframes', type=int, required=False, default=None, help='maximum number of processed frames')
    parser.add_argument('--batchsize', type=int, required=False, default=None, help='size of each batch that should go through the network')
    params = parser.parse_args()

    tick_t = timer()

    print('===> Loading model...')
    model = Net()
    state_dict = torch.load(params.model)
    model.load_state_dict(state_dict)

    def convert_frame(arg):
        return Image.fromarray(arg[:, :, :3], mode='RGB')

    print('===> Reading video...')
    video_reader = imageio.get_reader(params.src)
    input_fps = video_reader.get_meta_data()['fps']

    input_frames = [convert_frame(x) for x in video_reader]

    if params.maxframes is not None:
        input_frames = input_frames[:params.maxframes]

    batch_size = len(input_frames)
    if params.batchsize is not None and params.batchsize > 0:
        batch_size = min(params.batchsize, batch_size)

    print('===> Interpolating...')
    middle_frames = []
    n_baches = int(math.ceil(1.0 * len(input_frames) / batch_size))
    for i in range(n_baches):
        batch = input_frames[batch_size*i : batch_size*(i+1)]
        middle_frames += interpolate_batch(model, batch)

    print('===> Stitching frames...')
    output_frames = input_frames[:1]
    iters = len(input_frames) - 1
    for i in range(iters):
        frame1 = input_frames[i]
        frame2 = input_frames[i+1]
        middle = middle_frames[i]
        output_frames += [middle, frame2]
        print('{}/{} done'.format(i+1, iters))
    output_frames = [np.asarray(x) for x in output_frames]

    print('===> Saving video...')
    imageio.mimwrite(params.dest, output_frames, fps=(input_fps * 2))

    tock_t = timer()

    print("Done. Took ~{}s".format(round(tock_t - tick_t)))
