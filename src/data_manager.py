#
# KTH Royal Institute of Technology
#

import json
import random
import zipfile
import numpy as np
import cv2 as cv
from joblib import Parallel, delayed
from timeit import default_timer as timer
from torchvision.transforms.functional import crop as crop_image
from os.path import exists, join, basename, isdir
from os import makedirs, remove, listdir, rmdir
from six.moves import urllib
from PIL import Image

import src.config as config


############################################# UTILITIES #############################################

def load_img(file_path):
    """
    Reads an image from disk.
    :param file_path: Path to the image file
    :return: PIL.Image object
    """
    return Image.open(file_path).convert('RGB')


def is_image(file_path):
    """
    Checks if the specified file is an image
    :param file_path: Path to the candidate file
    :return: Whether or not the file is an image
    """
    return any(file_path.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_patch(patch):
    """
    Reads the three images of a patch from disk and returns them already cropped.
    :param patch: Dictionary containing the details of the patch
    :return: Tuple of PIL.Image objects corresponding to the patch
    """
    paths = (patch['left_frame'], patch['middle_frame'], patch['right_frame'])
    i, j = (patch['patch_i'], patch['patch_j'])
    imgs = [load_img(x) for x in paths]
    h, w = config.PATCH_SIZE
    return tuple(crop_image(x, i, j, h, w) for x in imgs)


def load_cached_patch(cached_patch):
    """
    Reads the three cached images of a patch from disk. Can only be used if the patches
    have been previously cached.
    :param cached_patch: Patch as a tuple (path_to_left, path_to_middle, path_to_right)
    :return: Tuple of PIL.Image objects corresponding to the patch
    """
    return tuple(load_img(x) for x in cached_patch)


############################################### DAVIS ###############################################

def _get_davis(dataset_dir):
    """
    Returns the local path to the DAVIS dataset, given its root directory. The dataset
    is downloaded if not found on disk.
    :param dataset_dir: Path to the dataset directory
    :return: Path to the DAVIS dataset
    """
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
    """
    Finds all images of the specified resolution from the DAVIS dataset. The found paths
    are returned as tuples of three elements.
    :param davis_dir: Path to the DAVIS dataset directory
    :param res: Resolution of the DAVIS images (either '480p' or '1080p')
    :return: List of paths as tuples (path_to_left, path_to_middle, path_to_right)
    """

    subdir = join(davis_dir, "JPEGImages/" + res)

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


########################################## PATCH EXTRACTION #########################################

def simple_flow(frame1, frame2):
    """
    Runs SimpleFlow given two consecutive frames.
    :param frame1: PIL.Image frame at time t
    :param frame2: PIL.Image frame at time t+1
    :return: Numpy array with the flow for each pixel. Shape is same as input
    """
    frame1 = pil_to_opencv(frame1)
    frame2 = pil_to_opencv(frame2)
    flow = cv.optflow.calcOpticalFlowSF(frame1, frame2, layers=3, averaging_block_size=2, max_flow=4)
    n = np.sum(1 - np.isnan(flow), axis=(0, 1))
    flow[np.isnan(flow)] = 0
    return np.linalg.norm(np.sum(flow, axis=(0, 1)) / n)


def pil_to_opencv(frame):
    open_cv_image = np.array(frame)
    # Convert RGB to BGR
    return open_cv_image[:, :, ::-1]


def is_jumpcut(frame1, frame2):
    """
    Detects a jumpcut between the two frames.
    :param frame1: PIL.Image frame at time t
    :param frame2: PIL.Image frame at time t+1
    :return: Whether or not there is a jumpcut between the two frames
    """
    # TODO: Implement
    return False


def _extract_patches_worker(tuples, max_per_frame=1, trials_per_tuple=100, min_avg_flow=0.0):
    """
    Extracts small patches from the original frames. The patches are selected to maximize
    their contribution to the training.
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

            avg_flow = simple_flow(left_patch, right_patch)
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


def _extract_patches(tuples, max_per_frame=1, trials_per_tuple=100, min_avg_flow=0.0, workers=0):
    """
    Spawns the specified number of workers running _extract_patches_worker().
    Call this with workers=0 to run on the current thread.
    """

    tick_t = timer()
    print('===> Extracting patches...')

    if workers != 0:
        parallel = Parallel(n_jobs=workers, backend='threading', verbose=5)
        tuples_per_job = len(tuples) // workers + 1
        result = parallel(
            delayed(_extract_patches_worker)(tuples[i:i + tuples_per_job], max_per_frame, trials_per_tuple) for i in
            range(0, len(tuples), tuples_per_job))
        patches = sum(result, [])
    else:
        patches = _extract_patches_worker(tuples, max_per_frame, trials_per_tuple)

    tock_t = timer()
    print("Done. Took ~{}s".format(round(tock_t - tick_t)))

    return patches


############################################### CACHE ###############################################

def get_cached_patches(dataset_dir=None):
    """
    Finds the cached patches (stored as images) from disk and returns their paths as a list of tuples
    :param dataset_dir: Path to the dataset folder
    :return: List of paths to patches as tuples (path_to_left, path_to_middle, path_to_right)
    """

    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR

    cache_dir = join(dataset_dir, 'cache')

    frame_paths = [join(cache_dir, x) for x in listdir(cache_dir)]
    frame_paths = [x for x in frame_paths if is_image(x)]
    frame_paths.sort()

    tuples = []

    for i in range(len(frame_paths) // 3):
        x1, t, x2 = frame_paths[i * 3], frame_paths[i * 3 + 1], frame_paths[i * 3 + 2]
        tuples.append((x1, t, x2))

    return tuples


def _cache_patches_worker(cache_dir, patches):
    """
    Writes to disk the specified patches as images.
    :param cache_dir: Path to the cache folder
    :param patches: List of patches
    """
    for p in patches:
        patch_id = str(random.randint(1e10, 1e16))
        frames = load_patch(p)
        for i in range(3):
            file_name = '{}_{}.jpg'.format(patch_id, i)
            frames[i].save(join(cache_dir, file_name))


def _cache_patches(cache_dir, patches, workers=0):
    """
    Spawns the specified number of workers running _cache_patches_worker().
    Call this with workers=0 to run on the current thread.
    """

    if exists(cache_dir):
        rmdir(cache_dir)

    makedirs(cache_dir)

    tick_t = timer()
    print('===> Caching patches...')

    if workers != 0:
        parallel = Parallel(n_jobs=workers, backend='threading', verbose=5)
        patches_per_job = len(patches) // workers + 1
        parallel(delayed(_cache_patches_worker)(cache_dir, patches[i:i + patches_per_job]) for i in
                 range(0, len(patches), patches_per_job))
    else:
        _cache_patches_worker(cache_dir, patches)

    tock_t = timer()
    print("Done. Took ~{}s".format(round(tock_t - tick_t)))


################################################ MAIN ###############################################

def prepare_dataset(dataset_dir=None, force_rebuild=False):
    """
    Performs all necessary operations to get the training dataset ready, such as
    selecting patches, caching the cropped versions if necessary, etc..
    :param dataset_dir: Path to the dataset folder
    :param force_rebuild: Whether or not the patches should be extracted again, even if a cached version exists on disk
    :return: List of patches
    """

    if dataset_dir is None:
        dataset_dir = config.DATASET_DIR

    workers = config.NUM_WORKERS
    json_path = join(dataset_dir, 'patches.json')
    cache_dir = join(dataset_dir, 'cache')

    if exists(json_path) and not force_rebuild:

        print('===> Patches already processed, reading from JSON...')
        with open(json_path) as f:
            patches = json.load(f)

        if config.CACHE_PATCHES and not exists(cache_dir):
            _cache_patches(cache_dir, patches, workers)

        return patches

    davis_dir = _get_davis(dataset_dir)
    tuples = _tuples_from_davis(davis_dir, res='1080p')

    patches = _extract_patches(tuples, max_per_frame=20, trials_per_tuple=20, workers=workers)

    # shuffle patches before writing to file
    random.shuffle(patches)

    print('===> Saving JSON...')
    with open(json_path, 'w') as f:
        json.dump(patches, f)

    if config.CACHE_PATCHES:
        _cache_patches(cache_dir, patches, workers)

    return patches
