#
# KTH Royal Institute of Technology
#

import torch.utils.data as data
import torch
from torchvision.transforms import CenterCrop
import numpy as np
from PIL import Image
import src.data_manager as data_manager
import src.config as config


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

class PatchDataset(data.Dataset):

    def __init__(self, patches, use_cache):
        super(PatchDataset, self).__init__()
        self.patches = patches
        self.crop = CenterCrop(config.CROP_SIZE)

        if use_cache:
            self.load_patch = data_manager.load_cached_patch
        else:
            self.load_patch = data_manager.load_patch

        print('Dataset ready with {} tuples.'.format(len(patches)))

    def __getitem__(self, index):
        frames = self.load_patch(self.patches[index])
        x1, target, x2 = (pil_to_tensor(self.crop(x)) for x in frames)
        input = torch.cat((x1, x2), dim=0)
        return input, target

    def __len__(self):
        return len(self.patches)

class TestDataset(data.Dataset):

    def __init__(self):
        super(TestDataset, self).__init__()
        print('TestDataset class not implemented!')

    def __getitem__(self, index):
        return None, None

    def __len__(self):
        return 0

def get_training_set():
    patches = data_manager.prepare_dataset()
    if config.CACHE_PATCHES:
        patches = data_manager.get_cached_patches()
    patches = patches[:config.MAX_TRAINING_SAMPLES]
    return PatchDataset(patches, config.CACHE_PATCHES)

def get_test_set():
    return TestDataset()
