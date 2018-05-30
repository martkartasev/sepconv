#
# KTH Royal Institute of Technology
#

import torch.utils.data as data
import torch
from torchvision.transforms import CenterCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
import numpy as np
import random
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
    x_np = pil_to_numpy(x_pil)
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

    def __init__(self, patches, use_cache, augment_data):
        super(PatchDataset, self).__init__()
        self.patches = patches
        self.crop = CenterCrop(config.CROP_SIZE)

        if augment_data:
            self.random_transforms = [RandomRotation((90, 90)), RandomVerticalFlip(1.0), RandomHorizontalFlip(1.0),
                                      (lambda x: x)]
            self.get_aug_transform = (lambda: random.sample(self.random_transforms, 1)[0])
        else:
            # Transform does nothing. Not sure if horrible or very elegant...
            self.get_aug_transform = (lambda: (lambda x: x))

        if use_cache:
            self.load_patch = data_manager.load_cached_patch
        else:
            self.load_patch = data_manager.load_patch

        print('Dataset ready with {} tuples.'.format(len(patches)))

    @staticmethod
    def random_temporal_order_swap(x1, x2):
        if random.random() <= config.RANDOM_TEMPORAL_ORDER_SWAP_PROB:
            return x2, x1
        else:
            return x1, x2

    def __getitem__(self, index):
        frames = self.load_patch(self.patches[index])
        aug_transform = self.get_aug_transform()
        x1, target, x2 = (pil_to_tensor(self.crop(aug_transform(x))) for x in frames)
        x1, x2, = self.random_temporal_order_swap(x1, x2)
        input = torch.cat((x1, x2), dim=0)
        return input, target

    def __len__(self):
        return len(self.patches)


class ValidationDataset(data.Dataset):

    def __init__(self, tuples):
        super(ValidationDataset, self).__init__()
        self.tuples = tuples
        self.crop = CenterCrop(config.CROP_SIZE)

    def __getitem__(self, index):
        frames = self.tuples[index]
        x1, target, x2 = (pil_to_tensor(self.crop(data_manager.load_img(x))) for x in frames)
        input = torch.cat((x1, x2), dim=0)
        return input, target

    def __len__(self):
        return len(self.tuples)


def get_training_set():
    patches = data_manager.prepare_dataset()
    if config.CACHE_PATCHES:
        patches = data_manager.get_cached_patches()
    patches = patches[:config.MAX_TRAINING_SAMPLES]
    return PatchDataset(patches, config.CACHE_PATCHES, config.AUGMENT_DATA)


def get_validation_set():
    davis_17_test = data_manager.get_davis_17_test(config.DATASET_DIR)
    tuples = data_manager.tuples_from_davis(davis_17_test, res='480p')
    n_samples = min(len(tuples), config.MAX_VALIDATION_SAMPLES)
    random_ = random.Random(42)
    tuples = random_.sample(tuples, n_samples)
    return ValidationDataset(tuples)


def get_visual_test_set():
    return data_manager.get_selected_davis(res='480p')
