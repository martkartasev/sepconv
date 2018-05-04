#
# KTH Royal Institute of Technology
#

import torch
import torch.nn as nn
import torch.optim as optim
from model import Net, CustomLoss
from data import get_training_set, get_test_set


if torch.cuda.is_available():
    print("===> CUDA available, proceeding with GPU...")
    device = torch.device("cuda")
else:
    print("===> No GPU found, proceeding with CPU...")
	device = torch.device("cpu")

# torch.manual_seed(42)

print('===> Loading datasets...')
train_set = get_training_set()
test_set = get_test_set()

print('===> Building model...')
model = Net().to(device)
criterion = CustomLoss()
optimizer = optim.Adam(model.parameters())

# ...

print('Done')
