#
# KTH Royal Institute of Technology
#

import torch.utils.data as data
import torch
from torchvision.transforms import Compose, CenterCrop, ToTensor
from os.path import exists, join, basename
from os import makedirs, remove, listdir
from six.moves import urllib
from PIL import Image
import zipfile

def load_img(file_path):
    img = Image.open(file_path).convert('RGB')
    x, _, _ = img.split()
    return x

class DatasetFromFolder(data.Dataset):

    def __init__(self, image_dir, input_transform, target_transform):
        super(DatasetFromFolder, self).__init__()

        paths = listdir(image_dir)
        tuples = []
        for i in range(len(paths) // 3):
            x1, t, x2 = paths[i*3], paths[i*3 +1], paths[i*3 +2]
            x1, t, x2 = join(image_dir, x1), join(image_dir, t), join(image_dir, x2)
            tuples.append((x1, t, x2))

        self.image_tuples = tuples
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        touple = self.image_tuples[index]
        x1, target, x2 = load_img(touple[0]), load_img(touple[1]), load_img(touple[2])

        if self.input_transform:
            x1 = self.input_transform(x1)
            x2 = self.input_transform(x2)
        if self.target_transform:
            target = self.target_transform(target)

        input = torch.tensor([x1, x2])
        return input, target

    def __len__(self):
        return len(self.image_tuples)

def download_davis(dest="./dataset"):

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
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])

def _target_transform(crop_size):
    return _input_transform(crop_size)

def get_training_set():
    root_dir = download_davis()
    jpegs_dir = join(root_dir, "JPEGImages/480p")
    crop_size = 128
    return DatasetFromFolder(jpegs_dir, _input_transform(crop_size), _target_transform(crop_size))

def get_test_set():
    root_dir = download_davis()
    jpegs_dir = join(root_dir, "JPEGImages/480p")
    crop_size = 128
    return DatasetFromFolder(jpegs_dir, _input_transform(crop_size), _target_transform(crop_size))
