#
# KTH Royal Institute of Technology
#

import imageio
import argparse
from os.path import join
from timeit import default_timer as timer
from PIL import Image


def extract_frames(video_path):

    def convert_frame(arg):
        return Image.fromarray(arg[:, :, :3], mode='RGB')

    video_reader = imageio.get_reader(video_path)
    fps = video_reader.get_meta_data().get('fps', None)
    frames = [convert_frame(x) for x in video_reader]

    return frames, fps


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Video Frame Extraction')
    parser.add_argument('--src', type=str, required=True, help='path to the video')
    parser.add_argument('--dest', type=str, required=True, help='path to the output directory')
    params = parser.parse_args()

    tick_t = timer()

    print('===> Extracting frames...')
    extracted_frames, _ = extract_frames(params.src)

    print('===> Writing results...')
    for i, frame in enumerate(extracted_frames):
        file_name = '{:05d}.jpg'.format(i)
        file_path = join(params.dest, file_name)
        frame.save(file_path)

    tock_t = timer()

    print("Done. Took ~{}s".format(round(tock_t - tick_t)))
