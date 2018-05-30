#
# KTH Royal Institute of Technology
#

import torch
import argparse
from PIL import Image
from timeit import default_timer as timer
from src.config import OUTPUT_1D_KERNEL_SIZE
from src.dataset import pil_to_tensor, numpy_to_pil


def _get_padding_modules(img_height, img_width):

    top = OUTPUT_1D_KERNEL_SIZE // 2
    bottom = OUTPUT_1D_KERNEL_SIZE // 2
    left = OUTPUT_1D_KERNEL_SIZE // 2
    right = OUTPUT_1D_KERNEL_SIZE // 2

    padding_width = left + img_width + right
    padding_height = top + img_height + bottom

    if padding_width % 128 != 0:
        padding_width = (padding_width // 128 + 1) * 128

    if padding_height % 128 != 0:
        padding_height = (padding_height // 128 + 1) * 128

    padding_width = padding_width - (left + img_width + right)
    padding_height = padding_height - (top + img_height + bottom)

    input_padding_module = torch.nn.ReplicationPad2d([left,  (right + padding_width),  top,  (bottom + padding_height)])
    output_padding_module = torch.nn.ReplicationPad2d([-left, -(right + padding_width), -top, -(bottom + padding_height)])

    return input_padding_module, output_padding_module


def interpolate_batch(model_, pil_frames):

    assert len(pil_frames) > 1, "Frames to be interpolated must be at least two"

    batch = []
    for i in range(len(pil_frames)-1):
        frame1 = pil_to_tensor(pil_frames[i])
        frame2 = pil_to_tensor(pil_frames[i+1])
        batch.append(torch.cat((frame1, frame2), dim=0))
    batch = torch.stack(batch, dim=0)

    frame_channels, frame_height, frame_width = batch[0].shape
    frame_channels /= 2
    assert frame_channels == 3, "Only frames with 3 channels are supported"

    # Generate the padding functions for the given input size
    input_pad, output_pad = _get_padding_modules(frame_height, frame_width)

    # Use CUDA if possible
    if torch.cuda.is_available():
        batch = batch.cuda()
        input_pad = input_pad.cuda()
        output_pad = output_pad.cuda()
        model_ = model_.cuda()

    # Apply input padding
    batch = input_pad(batch)

    # Run forward pass
    with torch.no_grad():
        output = model_(batch)

    # Apply output padding
    output = output_pad(output)

    # Get numpy representation of the output
    output = output.cpu().detach().numpy()

    output_pils = [numpy_to_pil(x) for x in output]
    return output_pils


def interpolate(model_, frame1, frame2):
    assert frame1.size == frame2.size, "Frames must be of the same size to be interpolated"
    frames = [frame1, frame2]
    return interpolate_batch(model_, frames)[0]


def interpolate_f(model_, path1, path2):
    frames = (Image.open(p).convert('RGB') for p in (path1, path2))
    return interpolate(model_, *frames)


if __name__ == '__main__':

    from src.model import Net

    parser = argparse.ArgumentParser(description='Frame Interpolation')
    parser.add_argument('--prev', type=str, required=True, help='path to frame at t-1')
    parser.add_argument('--succ', type=str, required=True, help='path to frame at t+1')
    parser.add_argument('--dest', type=str, required=True, help='output path of the resulting frame at t')
    parser.add_argument('--model', type=str, required=True, help='path of the trained model')
    params = parser.parse_args()

    tick_t = timer()

    print('===> Loading model...')
    model = Net.from_file(params.model)

    print('===> Interpolating...')
    frame_out = interpolate_f(model, params.prev, params.succ)

    print('===> Writing result...')
    frame_out.save(params.dest)

    tock_t = timer()

    print("Done. Took ~{}s".format(round(tock_t - tick_t)))
