#
# KTH Royal Institute of Technology
#

import torch
import argparse
import numpy as np
from PIL import Image
from timeit import default_timer as timer


def _img_trasform(img):
    # MUST mimic the behavior of pil_transform() in dataset.py
    return torch.from_numpy(np.rollaxis(np.asarray(img) / 255.0, 2)).float()

def interpolate(model, frame1, frame2):

    frame1 = _img_trasform(frame1)
    frame2 = _img_trasform(frame2)
    input = torch.cat((frame1, frame2), dim=0)

    if torch.cuda.is_available():
        input.cuda()
        model.cuda()

    output = model(input).cpu()
    output = output.detach().numpy()
    output *= 255.0
    output = output.clip(0, 255)
    frame_out = Image.fromarray(output, mode='RGB')
    return frame_out

def interpolate_f(model, path1, path2):
    frames = (Image.open(p).convert('RGB') for p in (path1, path2))
    return interpolate(model, *frames)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Video Frame Interpolation')
    parser.add_argument('--prev', type=str, required=True, help='path to frame at t-1')
    parser.add_argument('--succ', type=str, required=True, help='path to frame at t+1')
    parser.add_argument('--dest', type=str, required=True, help='output path of the resulting frame at t')
    parser.add_argument('--model', type=str, required=True, help='path of the trained model')
    params = parser.parse_args()

    tick_t = timer()

    print('===> Loading model...')
    model = torch.load(params.model)

    print('===> Interpolating...')
    frame_out = interpolate_f(model, params.prev, params.succ)

    print('===> Writing result...')
    frame_out.save(params.dest)

    tock_t = timer()

    print("Done. Took ~{}s".format(round(tock_t - tick_t)))
