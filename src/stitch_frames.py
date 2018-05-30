
import argparse
from timeit import default_timer as timer
from os.path import join
from os import listdir
from src.interpolate_video import _write_video
from src.data_manager import is_image, load_img


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Frame to video')
    parser.add_argument('--src', type=str, required=True, help='path to the video directory')
    parser.add_argument('--dest', type=str, required=True, help='path to the output file')
    parser.add_argument('--outputfps', type=int, required=False, default=None, help='frame-rate of the output')
    parser.add_argument('--dropframes', type=bool, required=False, default=False, help='if every other frame should be dropped')
    params = parser.parse_args()

    root_path = params.src

    tick_t = timer()

    frames = [join(root_path, x) for x in listdir(root_path)]
    frames = [x for x in frames if is_image(x)]
    frames.sort()

    frames = [load_img(x) for i, x in enumerate(frames)]

    if params.dropframes:
        frames = [x for i, x in enumerate(frames) if i%2 == 0]

    print('===> Writing results...')
    _write_video(params.dest, frames, params.outputfps)

    tock_t = timer()

    print("Done. Took ~{}s".format(round(tock_t - tick_t)))

