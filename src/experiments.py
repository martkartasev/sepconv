#
# KTH Royal Institute of Technology
#

import torch
from torchvision.transforms import CenterCrop
from os.path import join
from src.model import Net
from src.interpolate import interpolate
from src.extract_frames import extract_frames
from src.data_manager import load_img
from src.dataset import pil_to_tensor, get_validation_set
from src.utilities import psnr
from src.loss import ssim
import src.config as config


def test_metrics(model, video_path=None, frames=None, output_folder=None):

    if video_path is not None and frames is None:
        frames, _ = extract_frames(video_path)

    total_ssim = 0
    total_psnr = 0
    stride = 30
    iters = 1 + (len(frames) - 3) // stride

    triplets = []
    for i in range(iters):
        tup = (frames[i*stride], frames[i*stride + 1], frames[i*stride + 2])
        triplets.append(tup)

    iters = len(triplets)

    for i in range(iters):
        x1, gt, x2 = triplets[i]
        pred = interpolate(model, x1, x2)
        if output_folder is not None:
            frame_path = join(output_folder, f'wiz_{i}.jpg')
            pred.save(frame_path)
        gt = pil_to_tensor(gt)
        pred = pil_to_tensor(pred)
        total_ssim += ssim(pred, gt).item()
        total_psnr += psnr(pred, gt).item()
        print(f'#{i+1}/{iters} done')

    avg_ssim = total_ssim / iters
    avg_psnr = total_psnr / iters

    print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr}')


def test_wiz(model, output_folder=None):
    video_path = '/project/videos/see_you_again_540.mp4'
    test_metrics(model, video_path=video_path, output_folder=output_folder)


def test_on_validation_set(model, validation_set=None):

    if validation_set is None:
        validation_set = get_validation_set()

    total_ssim = 0
    total_psnr = 0
    iters = len(validation_set.tuples)

    crop = CenterCrop(config.CROP_SIZE)

    for i, tup in enumerate(validation_set.tuples):
        x1, gt, x2, = [crop(load_img(p)) for p in tup]
        pred = interpolate(model, x1, x2)
        gt = pil_to_tensor(gt)
        pred = pil_to_tensor(pred)
        total_ssim += ssim(pred, gt).item()
        total_psnr += psnr(pred, gt).item()
        print(f'#{i+1} done')

    avg_ssim = total_ssim / iters
    avg_psnr = total_psnr / iters

    print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr}')


def test_linear_interp(validation_set=None):

    if validation_set is None:
        validation_set = get_validation_set()

    total_ssim = 0
    total_psnr = 0
    iters = len(validation_set.tuples)

    crop = CenterCrop(config.CROP_SIZE)

    for tup in validation_set.tuples:
        x1, gt, x2, = [pil_to_tensor(crop(load_img(p))) for p in tup]
        pred = torch.mean(torch.stack((x1, x2), dim=0), dim=0)
        total_ssim += ssim(pred, gt).item()
        total_psnr += psnr(pred, gt).item()

    avg_ssim = total_ssim / iters
    avg_psnr = total_psnr / iters

    print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr}')


def test_all():

    print('===> Loading pure L1...')
    # pure_l1 = Net.from_file('./trained_models/last_pure_l1.pth')

    print('===> Testing latest pure L1...')
    # test_on_validation_set(pure_l1)
    print('avg_ssim: 0.8197908288240433, avg_psnr: 29.126618137359618')

    print('===> Testing linear interp...')
    # test_linear_interp()
    print('avg_ssim: 0.6868560968339443, avg_psnr: 26.697076902389526')

    print('===> Loading best models...')
    # best_model_qualitative = Net.from_file('./trained_models/best_model_qualitative.pth')
    # best_model_quantitative = Net.from_file('./trained_models/best_model_quantitative.pth')

    print('===> Testing Wiz (qualitative)...')
    # test_wiz(best_model_qualitative, output_folder='/project/exp/wiz_qual/')
    print('avg_ssim: 0.9658980375842044, avg_psnr: 37.27564642554835')

    print('===> Testing Wiz (quantitative)...')
    # test_wiz(best_model_quantitative, output_folder='/project/exp/wiz_quant/')
    print('avg_ssim: 0.9638479389642415, avg_psnr: 36.52394056822124')

if __name__ == '__main__':
    test_all()
