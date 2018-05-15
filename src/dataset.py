#
# KTH Royal Institute of Technology
#

import torch.utils.data as data
import torch
import numpy as np
from torchvision.transforms import CenterCrop
from os.path import exists, join, basename, isdir
from os import makedirs, remove, listdir
from six.moves import urllib
from PIL import Image
import src.config as config
import zipfile

def load_img(file_path):
    return Image.open(file_path).convert('RGB')

def is_image(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def pil_to_numpy(x_pil):
    """
    :param x_pil: PIL.Image object
    :return: Normalized numpy array of shape (channels, height, width)
    """
    # Channels are the third dim of a PIL.Image,
    # but we want to be able to index it by channel first,
    # so we use np.rollaxis to get an array of shape (3, h, w)
    return np.rollaxis(np.asarray(x_pil) / 255.0, 2)

def pil_to_tensor(x_pil):
    """
    :param x_pil: PIL.Image object
    :return: Normalized torch tensor of shape (channels, height, width)
    """
    x_np =pil_to_numpy(x_pil)
    return torch.from_numpy(x_np).float()

def numpy_to_pil(x_np):
    """
    :param x_np: Image as a numpy array of shape (channels, height, width)
    :return: PIL.Image object
    """
    x_np = x_np.copy()
    x_np *= 255.0
    x_np = x_np.clip(0, 255)
    # PIL.Image wants the channel as the last dimension
    x_np = np.rollaxis(x_np, 0, 3).astype(np.uint8)
    return Image.fromarray(x_np, mode='RGB')

class DatasetFromFolder(data.Dataset):

    def __init__(self, root_dir, input_transform, target_transform):
        super(DatasetFromFolder, self).__init__()

        video_dirs = [join(root_dir, x) for x in listdir(root_dir)]
        video_dirs = [x for x in video_dirs if isdir(x)]

        tuples = []
        for video_dir in video_dirs:

            frame_paths = [join(video_dir, x) for x in listdir(video_dir)]
            frame_paths = [x for x in frame_paths if is_image(x)]

            for i in range(len(frame_paths) // 3):
                x1, t, x2 = frame_paths[i*3], frame_paths[i*3 +1], frame_paths[i*3 +2]
                tuples.append((x1, t, x2))

        if config.MAX_TRAINING_SAMPLES is not None:
            tuples = tuples[:config.MAX_TRAINING_SAMPLES]

        self.image_tuples = tuples
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        tup = self.image_tuples[index]
        x1, target, x2 = (load_img(x) for x in tup)

        if self.input_transform:
            x1 = self.input_transform(x1)
            x2 = self.input_transform(x2)

        if self.target_transform:
            target = self.target_transform(target)

        x1 = pil_to_tensor(x1)
        x2 = pil_to_tensor(x2)
        target = pil_to_tensor(target)

        input = torch.cat((x1, x2), dim=0)
        return input, target

    def __len__(self):
        return len(self.image_tuples)

def download_davis(dest=None):

    if dest is None:
        dest = config.DATASET_DIR

    output_dir = join(dest, "DAVIS")

    if not exists(output_dir):

        if not exists(dest):
            makedirs(dest)

        url = "https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip"

        print("===> Downloading DAVIS...")
        response = urllib.request.urlopen(url)
        zip_path = join(dest, basename(url))
        with open(zip_path, 'wb') as f:
            f.write(response.read())

        zip_ref = zipfile.ZipFile(zip_path, 'r')

        print("===> Extracting data...")
        zip_ref.extractall(dest)
        zip_ref.close()

        # Remove downloaded zip file
        remove(zip_path)

    return output_dir

def _input_transform(crop_size):
    return CenterCrop(crop_size)

def _target_transform(crop_size):
    return _input_transform(crop_size)

def get_training_set():
    root_dir = download_davis()
    jpegs_dir = join(root_dir, "JPEGImages/480p")
    crop_size = config.CROP_SIZE
    return DatasetFromFolder(jpegs_dir, _input_transform(crop_size), _target_transform(crop_size))

def get_test_set():
    root_dir = download_davis()
    jpegs_dir = join(root_dir, "JPEGImages/480p")
    crop_size = config.CROP_SIZE
    return DatasetFromFolder(jpegs_dir, _input_transform(crop_size), _target_transform(crop_size))
