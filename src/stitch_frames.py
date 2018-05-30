#
# KTH Royal Institute of Technology
#

import argparse
from timeit import default_timer as timer
from os.path import join
from os import listdir
from src.utilities import write_video
from src.data_manager import is_image, load_img


def stitch_frames(src_path, dest_path, output_fps=None, drop_frames=False):

    tick_t = timer()

    frames = [join(src_path, x) for x in listdir(src_path)]
    frames = [x for x in frames if is_image(x)]
    frames.sort()

    print('===> Loading frames...')
    if drop_frames:
        frames = [load_img(x) for i, x in enumerate(frames) if i % 2 == 0]
    else:
        frames = [load_img(x) for i, x in enumerate(frames)]

    print('===> Writing results...')
    write_video(dest_path, frames, output_fps)

    tock_t = timer()
    print("Done. Took ~{}s".format(round(tock_t - tick_t)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Frames to video file')
    parser.add_argument('--src', dest='src_path', type=str, required=True, help='path to the directory containing the frames')
    parser.add_argument('--dest', dest='dest_path', type=str, required=True, help='path to the output file')
    parser.add_argument('--outputfps', dest='output_fps', type=int, required=False, default=None, help='frame-rate of the output')
    parser.add_argument('--dropframes', dest='drop_frames', type=bool, required=False, default=False, help='whether or not every other frame should be dropped')
    stitch_frames(**vars(parser.parse_args()))
