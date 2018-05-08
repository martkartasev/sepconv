#
# KTH Royal Institute of Technology
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import src.config as config
from src.model import Net, CustomLoss
from src.dataset import get_training_set, get_test_set


if config.ALWAYS_CPU:
    print("===> ALWAYS_CPU is True, proceeding with CPU...")
    device = torch.device("cpu")
elif torch.cuda.is_available():
    print("===> CUDA available, proceeding with GPU...")
    device = torch.device("cuda")
else:
    print("===> No GPU found, proceeding with CPU...")
    device = torch.device("cpu")

if config.SEED is not None:
    torch.manual_seed(config.SEED)


print('===> Loading datasets...')
train_set = get_training_set()
test_set = get_test_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE, shuffle=False)


print('===> Building model...')
model = Net().to(device)
l1_loss = nn.L1Loss()
optimizer = optim.Adam(model.parameters())

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = l1_loss(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


for epoch in range(1, config.EPOCHS + 1):
    train(epoch)

print('Done')
