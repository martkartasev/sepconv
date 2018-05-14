#
# KTH Royal Institute of Technology
#

import imageio
import argparse
import torch
import numpy as np
#from joblib import Parallel, delayed
from torchvision.transforms import CenterCrop
from timeit import default_timer as timer
from PIL import Image
from src.interpolate import interpolate


if __name__ == '__main__':

    from src.model import Net

    parser = argparse.ArgumentParser(description='Video Frame Interpolation')
    parser.add_argument('--src', type=str, required=True, help='path to the video')
    parser.add_argument('--dest', type=str, required=True, help='output path of the resulting video')
    parser.add_argument('--model', type=str, required=True, help='path of the trained model')
    # params = parser.parse_args()

    class params():
        model = '/Users/carlo/Desktop/model_epoch_10.pth'
        src = '/Users/carlo/Downloads/giphy.mp4'
        dest = '/Users/carlo/Desktop/video_out.mp4'

    tick_t = timer()

    print('===> Loading model...')
    model = Net()
    state_dict = torch.load(params.model)
    model.load_state_dict(state_dict)

    print('===> Reading video...')

    input_frames = imageio.mimread(params.src)

    center_crop = CenterCrop(256)

    def convert_frame(arg):
        return center_crop(Image.fromarray(arg[:, :, :3], mode='RGB'))

    input_frames = [convert_frame(x) for x in input_frames]

    input_frames = input_frames[:10]

    print('===> Interpolating...')
    output_frames = input_frames[:1]
    iters = len(input_frames)-1

    # middle_frames = Parallel(n_jobs=2, verbose=5, backend="threading")(delayed(interpolate)(model, input_frames[i], input_frames[i + 1]) for i in range(iters))

    for i in range(iters):
        frame1 = input_frames[i]
        frame2 = input_frames[i+1]
        # middle = middle_frames[i]
        middle = interpolate(model, frame1, frame2)
        output_frames += [middle, frame2]
        print('{}/{} done'.format(i+1, iters))

    output_frames = [np.asarray(x) for x in output_frames]

    print('===> Writing result...')
    imageio.mimwrite(params.dest, output_frames, fps=30)

    tock_t = timer()

    print("Done. Took ~{}s".format(round(tock_t - tick_t)))
