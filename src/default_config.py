#
# KTH Royal Institute of Technology
#

#
# This file is for reference only and should *not* me modified.
# To change the configuration, create a new config.py file
# and import this one in it as 'from src.default_config import *'
#

# The size of the input images to be fed to the network during training.
CROP_SIZE: int = 128

# The size of the patches to be extracted from the datasets
PATCH_SIZE = (150, 150)

# Whether or not we should store the patches produced by the data manager
CACHE_PATCHES: bool = False

# Number of epochs used for training
EPOCHS: int = 10

# Kernel size of the custom Separable Convolution layer
OUTPUT_1D_KERNEL_SIZE: int = 51

# The batch size used for mini batch gradient descent
BATCH_SIZE: int = 100

# Upper limit on the number of samples used for training
MAX_TRAINING_SAMPLES: int = 500_000

# Upper limit on the number of samples used for validation
MAX_VALIDATION_SAMPLES: int = 100

# Number of workers of the torch.utils.data.DataLoader AND of the data manager
# Set this to 0 to work on the main thread
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

# Force model to use the slow Separable Convolution implementation even if CUDA is available
ALWAYS_SLOW_SEP_CONV: bool = False

# Whether or not we should run the validation set on the model at each epoch
VALIDATION_ENABLED: bool = False

# Whether or not we should run the visual test set on the model at each epoch
VISUAL_TEST_ENABLED: bool = False

# Whether or not the data should be augmented with random transformations
AUGMENT_DATA: bool = True

# Probability of performing the random temporal order swap of the two input frames
RANDOM_TEMPORAL_ORDER_SWAP_PROB: float = 0.5

# Start from pre-trained model (path)
START_FROM_EXISTING_MODEL = None

# One of {"l1", "vgg", "ssim"}
LOSS: str = "l1"

VGG_FACTOR: float = 1.0
