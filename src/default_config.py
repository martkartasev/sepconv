#
# KTH Royal Institute of Technology
#

#
# This file is for reference only and should *not* me modified.
# To change the configuration, create a new config.py file
# and import this one in it as 'from default_config import *'
#

# The size of the input images to be fed to the network.
# Image files larger than this size will be cropped (around the center).
CROP_SIZE: int = 128

# TODO: Add description
EPOCHS: int = 10

# TODO: Add description
# TODO: Implement
OUTPUT_1D_KERNEL_SIZE: int = 51

# TODO: Add description
BATCH_SIZE: int = 100

# TODO: Add description
MAX_TRAINING_SAMPLES: int = 10_000

# Number of workers of the torch.utils.data.DataLoader
# Set this to 0 to force the DataLoader to work on the main thread
NUM_WORKERS: int = 0

# Random seed fed to torch
SEED: int = None

# Path to the dataset directory
DATASET_DIR = './dataset'

# Force torch to run on CPU even if CUDA is available
ALWAYS_CPU: bool = False

# Path to the outout directory where the model checkpoins should be stored
OUTPUT_DIR: str = './out'

# Whether or not the model parameters should be written to disk at each epoch
SAVE_CHECKPOINS: bool = False
