#
# KTH Royal Institute of Technology
#

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
from os.path import exists, join as join_paths
from os import makedirs, link, remove
from timeit import default_timer as timer
import src.config as config
from src import loss
from src.data_manager import load_img
from src.model import Net
from src.dataset import get_training_set, get_validation_set, get_visual_test_set, pil_to_tensor
from src.interpolate import interpolate
from src.utilities import psnr

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
validation_set = get_validation_set()
visual_test_set = get_visual_test_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE,
                                  shuffle=True)
validation_data_loader = DataLoader(dataset=validation_set, num_workers=config.NUM_WORKERS,
                                    batch_size=config.BATCH_SIZE, shuffle=False)

if config.START_FROM_EXISTING_MODEL is not None:
    print(f'===> Loading pre-trained model: {config.START_FROM_EXISTING_MODEL}')
    model = Net.from_file(config.START_FROM_EXISTING_MODEL)
else:
    print('===> Building model...')
    model = Net()

model.to(device)

if config.LOSS == "l1":
    loss_function = nn.L1Loss()
elif config.LOSS == "vgg":
    loss_function = loss.VggLoss()
elif config.LOSS == "ssim":
    loss_function = loss.SsimLoss()
elif config.LOSS == "l1+vgg":
    loss_function = loss.CombinedLoss()
else:
    raise ValueError(f"Unknown loss: {config.LOSS}")

optimizer = optim.Adamax(model.parameters(), lr=0.001)

board_writer = SummaryWriter()

# ----------------------------------------------------------------------

def train(epoch):
    print("===> Training...")
    before_pass = [p.data.clone() for p in model.parameters()]
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        print('Forward pass...')
        output = model(input)

        loss_ = loss_function(output, target)

        print('Computing gradients...')
        loss_.backward()

        print('Gradients ready.')
        optimizer.step()

        loss_val = loss_.item()
        epoch_loss += loss_val

        board_writer.add_scalar('data/iter_training_loss', loss_val, iteration)
        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss_val))

    weight_l2s = 0
    weight_diff_l2s = 0
    gradient_l2s = 0
    for i, p in enumerate(model.parameters()):
        weight_l2s += p.data.norm(2)
        weight_diff_l2s += (p.data - before_pass[i]).norm(2)
        gradient_l2s += p.grad.norm(2)
    board_writer.add_scalar('data/epoch_weight_l2', weight_l2s, epoch)
    board_writer.add_scalar('data/epoch_weight_change_l2', weight_diff_l2s, epoch)
    board_writer.add_scalar('data/epoch_gradient_l2', gradient_l2s, epoch)
    epoch_loss /= len(training_data_loader)
    board_writer.add_scalar('data/epoch_training_loss', epoch_loss, epoch)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss))


def save_checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    model_out_path = join_paths(config.OUTPUT_DIR, model_out_path)
    model_latest_path = join_paths(config.OUTPUT_DIR, 'model_epoch_latest.pth')
    if not exists(config.OUTPUT_DIR):
        makedirs(config.OUTPUT_DIR)
    torch.save(model.cpu().state_dict(), model_out_path)
    if exists(model_latest_path):
        remove(model_latest_path)
    link(model_out_path, model_latest_path)
    print("Checkpoint saved to {}".format(model_out_path))
    if device.type != 'cpu':
        model.cuda()


def validate(epoch):
    print("===> Running validation...")
    ssmi = loss.SsimLoss()
    valid_loss, valid_ssmi, valid_psnr = 0, 0, 0
    iters = len(validation_data_loader)
    with torch.no_grad():
        for batch in validation_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            output = model(input)
            valid_loss += loss_function(output, target).item()
            valid_ssmi -= ssmi(output, target).item()
            valid_psnr += psnr(output, target).item()
    valid_loss /= iters
    valid_ssmi /= iters
    valid_psnr /= iters
    board_writer.add_scalar('data/epoch_validation_loss', valid_loss, epoch)
    board_writer.add_scalar('data/epoch_ssmi', valid_ssmi, epoch)
    board_writer.add_scalar('data/epoch_psnr', valid_psnr, epoch)
    print("===> Validation loss: {:.4f}".format(valid_loss))


def visual_test(epoch):
    print("===> Running visual test...")
    for i, tup in enumerate(visual_test_set):
        result = interpolate(model, load_img(tup[0]), load_img(tup[2]))
        result = pil_to_tensor(result)
        tag = 'data/visual_test_{}'.format(i)
        board_writer.add_image(tag, result, epoch)


# ----------------------------------------------------------------------

tick_t = timer()

for epoch in range(1, config.EPOCHS + 1):
    train(epoch)
    if config.SAVE_CHECKPOINS:
        save_checkpoint(epoch)
    if config.VALIDATION_ENABLED:
        validate(epoch)
    if config.VISUAL_TEST_ENABLED:
        visual_test(epoch)

tock_t = timer()

print("Done. Took ~{}s".format(round(tock_t - tick_t)))

board_writer.close()
