#
# KTH Royal Institute of Technology
#

import torch.utils.data as data
import torch
import numpy as np
from torchvision.transforms import Compose, CenterCrop, ToTensor
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

def pil_transform(x):
    """
    :param x: PIL.Image object
    :return: Normalized torch tensor of shape (channels, height, width)
    """
    # Channels are the third dim of a PIL.Image,
    # but we want to be able to index it by channel first,
    # so we use np.rollaxis to get an array of shape (3, h, w)
    return torch.from_numpy(np.rollaxis(np.asarray(x) / 255.0, 2)).float()

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

        x1 = pil_transform(x1)
        x2 = pil_transform(x2)
        target = pil_transform(target)

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
