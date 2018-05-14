#
# KTH Royal Institute of Technology
#

import torch
from torchvision.transforms import CenterCrop
import argparse
import numpy as np
from PIL import Image
from timeit import default_timer as timer
from src.config import CROP_SIZE


_center_crop = CenterCrop(CROP_SIZE)

def _img_trasform(img):
    img = _center_crop(img)
    # MUST mimic the behavior of pil_transform() in dataset.py
    return torch.from_numpy(np.rollaxis(np.asarray(img) / 255.0, 2)).float()

def interpolate(model, frame1, frame2):

    frame1 = _img_trasform(frame1)
    frame2 = _img_trasform(frame2)
    _input = torch.cat((frame1, frame2), dim=0)

    # _input must be 4D (dim=0 being the index within the batch)
    _input = _input.view(1, _input.size(0), _input.size(1), _input.size(2))

    if torch.cuda.is_available():
        _input.cuda()
        model.cuda()

    output = model(_input).cpu()
    output = output.detach().numpy()[0]
    output *= 255.0
    output = output.clip(0, 255)

    # PIL.Image wants the channel as the last dimension
    output = np.rollaxis(output, 0, 3)
    frame_out = Image.fromarray(output, mode='RGB')

    return frame_out

def interpolate_f(model, path1, path2):
    frames = (Image.open(p).convert('RGB') for p in (path1, path2))
    return interpolate(model, *frames)


if __name__ == '__main__':

    from src.model import Net

    parser = argparse.ArgumentParser(description='Video Frame Interpolation')
    parser.add_argument('--prev', type=str, required=True, help='path to frame at t-1')
    parser.add_argument('--succ', type=str, required=True, help='path to frame at t+1')
    parser.add_argument('--dest', type=str, required=True, help='output path of the resulting frame at t')
    parser.add_argument('--model', type=str, required=True, help='path of the trained model')
    params = parser.parse_args()

    tick_t = timer()

    print('===> Loading model...')
    model = Net()
    state_dict = torch.load(params.model)
    model.load_state_dict(state_dict)

    print('===> Interpolating...')
    frame_out = interpolate_f(model, params.prev, params.succ)

    print('===> Writing result...')
    frame_out.save(params.dest)

    tock_t = timer()

    print("Done. Took ~{}s".format(round(tock_t - tick_t)))
