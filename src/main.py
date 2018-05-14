#
# KTH Royal Institute of Technology
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from os.path import exists, join as join_paths
from os import makedirs
from timeit import default_timer as timer
import src.config as config
from src.model import Net, CustomLoss
from src.dataset import get_training_set, get_test_set

from torch.autograd import Variable


# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------

print('===> Loading datasets...')
train_set = get_training_set()
test_set = get_test_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE, shuffle=False)

print('===> Building model...')
model = Net().to(device)
model_params = model.parameters()
l1_loss = nn.L1Loss()
optimizer = optim.Adamax(model_params, lr=0.001)


# ----------------------------------------------------------------------

def detach_all(arg):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(arg) == Variable:
        arg.detach_() # Variable(arg.data)
    else:
        for v in arg:
            detach_all(v)

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        detach_all(model_params)

        optimizer.zero_grad()

        print('Forward pass...')
        output = model(input)

        loss = l1_loss(output, target)
        epoch_loss += loss.item()

        print('Computing gradients...')
        loss.backward()

        print('Gradients ready.')
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def test():
    error = 0.0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            output = model(input)
            # TODO: compute error
            error = 0.0
    print("===> Test error: {:.4f}".format(error))

def save_checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    model_out_path = join_paths(config.OUTPUT_DIR, model_out_path)
    if not exists(config.OUTPUT_DIR):
        makedirs(config.OUTPUT_DIR)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# ----------------------------------------------------------------------

tick_t = timer()

for epoch in range(1, config.EPOCHS + 1):
    train(epoch)
    if config.SAVE_CHECKPOINS:
        save_checkpoint(epoch)
    if config.TEST_ENABLED:
        test()

tock_t = timer()

print("Done. Took ~{}s".format(round(tock_t - tick_t)))

#
# In order to interpolate two frames and write the output as an image file:
#
# model.interpolate_f(
#   '/path/to/frame_00001.jpg',
#   '/path/to/frame_00003.jpg'
# ).save('/path/to/frame_00002_star.jpg')
#
