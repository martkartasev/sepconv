#
# KTH Royal Institute of Technology
#

import json
import random
import zipfile
import numpy as np
from joblib import Parallel, delayed
from timeit import default_timer as timer
from torchvision.transforms.functional import crop as crop_image
from os.path import exists, join, basename, isdir
from os import makedirs, remove, listdir
from six.moves import urllib
from PIL import Image
import src.config as config


def load_img(file_path):
    """
    :param file_path: Path to the image file
    :return: PIL.Image object
    """
    return Image.open(file_path).convert('RGB')

def is_image(file_path):
    """
    :param file_path: Path to the candidate file
    :return: Whether or not the file is a usable image
    """
    return any(file_path.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def _get_davis(dataset_dir):

    davis_dir = join(dataset_dir, "DAVIS")

    if not exists(davis_dir):

        if not exists(dataset_dir):
            makedirs(dataset_dir)

        url = "https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip"

        print("===> Downloading DAVIS...")
        response = urllib.request.urlopen(url)
        zip_path = join(dataset_dir, basename(url))
        with open(zip_path, 'wb') as f:
            f.write(response.read())

        zip_ref = zipfile.ZipFile(zip_path, 'r')

        print("===> Extracting data...")
        zip_ref.extractall(dataset_dir)
        zip_ref.close()

        # Remove downloaded zip file
        remove(zip_path)

    return davis_dir

def _tuples_from_davis(davis_dir, res='480p'):

    subdir = join(davis_dir, "JPEGImages/"+res)

    video_dirs = [join(subdir, x) for x in listdir(subdir)]
    video_dirs = [x for x in video_dirs if isdir(x)]

    tuples = []
    for video_dir in video_dirs:

        frame_paths = [join(video_dir, x) for x in listdir(video_dir)]
        frame_paths = [x for x in frame_paths if is_image(x)]
        frame_paths.sort()

        for i in range(len(frame_paths) // 3):
            x1, t, x2 = frame_paths[i * 3], frame_paths[i * 3 + 1], frame_paths[i * 3 + 2]
            tuples.append((x1, t, x2))

    return tuples


def _extract_patches(tuples, max_per_frame=1, trials_per_tuple=100, min_avg_flow=0.0):
    """
    :param tuples: List of tuples containing the input frames as (left, middle, right)
    :param max_per_frame: Maximum number of patches that can be extracted from a frame
    :param trials_per_tuple: Number of random crops to test for each tuple
    :param min_avg_flow: Minimum average optical flow for a patch to be selected
    :return: List of dictionaries representing each patch
    """

    patch_h, patch_w = config.PATCH_SIZE
    n_tuples = len(tuples)
    all_patches = []

    for tup_index in range(n_tuples):
        tup = tuples[tup_index]

        left, middle, right = (load_img(x) for x in tup)
        img_w, img_h = left.size

        if is_jumpcut(left, middle) or is_jumpcut(middle, right):
            continue

        selected_patches = []

        for _ in range(trials_per_tuple):

            i = random.randint(0, img_h - patch_h)
            j = random.randint(0, img_w - patch_w)

            left_patch = crop_image(left, i, j, patch_h, patch_w)
            right_patch = crop_image(right, i, j, patch_h, patch_w)
            # middle_patch = crop_image(middle, i, j, patch_h, patch_w)

            avg_flow = np.mean(simple_flow(left_patch, right_patch))
            if avg_flow < min_avg_flow:
                continue

            selected_patches.append({
                "left_frame": tup[0],
                "middle_frame": tup[1],
                "right_frame": tup[2],
                "patch_i": i,
                "patch_j": j,
                "avg_flow": avg_flow
            })

        sorted(selected_patches, key=lambda x: x['avg_flow'], reverse=True)
        all_patches += selected_patches[:max_per_frame]
        # print("===> Tuple {}/{} ready.".format(tup_index+1, n_tuples))

    return all_patches


def simple_flow(frame1, frame2):
    """
    :param frame1: PIL.Image frame at time t
    :param frame2: PIL.Image frame at time t+1
    :return: Numpy array with the flow for each pixel. Shape is same as input
    """
    # TODO: Implement
    return np.zeros(frame1.size)

def is_jumpcut(frame1, frame2):
    """
    :param frame1: PIL.Image frame at time t
    :param frame2: PIL.Image frame at time t+1
    :return: Whether or not there is a jumpcut between the two frames
    """
    # TODO: Implement
    return False

def load_patch(patch):
    """
    :param patch: Dictionary containing the details of the patch
    :return: PIL.Image object corresponding to the patch
    """
    paths = (patch['left_frame'], patch['middle_frame'], patch['right_frame'])
    i, j = (patch['patch_i'], patch['patch_j'])
    imgs = [load_img(x) for x in paths]
    h, w = config.PATCH_SIZE
    return tuple(crop_image(x, i, j, h, w) for x in imgs)

def prepare_dataset(dataset_dir=None, force_rebuild=False):
    """
    :param dataset_dir: Path to the dataset folder
    :param force_rebuild: Whether or not the patches should be extracted again, even if a cached version exists on disk
    :return: List of patches
    """

    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR

    json_path = join(dataset_dir, 'patches.json')

    if exists(json_path) and not force_rebuild:
        print('===> Patches already processed, reading from JSON...')
        with open(json_path) as f:
            return json.load(f)

    davis_dir = _get_davis(dataset_dir)
    tuples = _tuples_from_davis(davis_dir, res='1080p')

    workers = config.NUM_WORKERS
    max_per_frame = 20
    trials_per_tuple = 20

    tick_t = timer()

    print('===> Extracting patches...')
    if workers != 0:
        parallel = Parallel(n_jobs=workers, backend='threading', verbose=5)
        tuples_per_job = len(tuples)//workers +1
        result = parallel(delayed(_extract_patches)(tuples[i:i+tuples_per_job], max_per_frame, trials_per_tuple) for i in range(0,len(tuples), tuples_per_job))
        patches = sum(result, [])
    else:
        patches = _extract_patches(tuples, max_per_frame, trials_per_tuple)

    tock_t = timer()

    print("Done. Took ~{}s".format(round(tock_t - tick_t)))

    with open(json_path, 'w') as f:
        json.dump(patches, f)

    return patches
