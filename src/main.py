#
# KTH Royal Institute of Technology
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Net, CustomLoss
from dataset import get_training_set, get_test_set


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
training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=100, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=100, shuffle=False)


print('===> Building model...')
model = Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# ...

print('Done')
