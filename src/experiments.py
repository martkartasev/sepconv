#
# KTH Royal Institute of Technology
#

import torch
from os.path import join
from src.model import Net
from src.interpolate import interpolate
from src.extract_frames import extract_frames
from src.data_manager import load_img
from src.dataset import pil_to_tensor, get_validation_set
from src.utilities import psnr
from src.loss import ssim


def test_metrics(model, video_path=None, frames=None, output_folder=None):

    if video_path is not None and frames is None:
        frames, _ = extract_frames(video_path)

    total_ssim = 0
    total_psnr = 0
    iters = 1 + (len(frames) - 3) // 2

    for i in range(iters):
        x1, gt, x2 = frames[i*2], frames[i*2 + 1], frames[i*2 + 2]
        pred = interpolate(model, x1, x2)
        if output_folder is not None:
            frame_path = join(output_folder, f'wiz_{i}.jpg')
            pred.save(frame_path)
        gt = pil_to_tensor(gt)
        pred = pil_to_tensor(pred)
        total_ssim += ssim(pred, gt).item()
        total_psnr += psnr(pred, gt).item()
        print(f'#{i+1} done')

    avg_ssim = total_ssim / iters
    avg_psnr = total_psnr / iters

    print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr}')


def load_model(path):
    model = Net()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model


def test_wiz(model, output_folder=None):
    video_path = '/project/videos/see_you_again_540.mp4'
    test_metrics(model, video_path=video_path, output_folder=output_folder)


def test_on_validation_set(model, validation_set=None):

    if validation_set is None:
        validation_set = get_validation_set()

    total_ssim = 0
    total_psnr = 0
    iters = len(validation_set.tuples)

    for tup in validation_set.tuples:
        x1, gt, x2, = [load_img(p) for p in tup]
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

    for tup in validation_set.tuples:
        x1, gt, x2, = [pil_to_tensor(load_img(p)) for p in tup]
        gt = pil_to_tensor(gt)
        pred = torch.mean(torch.stack((x1, x2), dim=0), dim=0)
        total_ssim += ssim(pred, gt).item()
        total_psnr += psnr(pred, gt).item()

    avg_ssim = total_ssim / iters
    avg_psnr = total_psnr / iters

    print(f'avg_ssim: {avg_ssim}, avg_psnr: {avg_psnr}')


def test_all():

    print('===> Loading pure L1...')
    pure_l1 = load_model('./trained_models/last_pure_l1.pth')

    print('===> Testing latest pure L1...')
    test_on_validation_set(pure_l1)

    print('===> Testing linear interp...')
    test_linear_interp()

    print('===> Loading best models...')
    best_model_qualitative = load_model('./trained_models/best_model_qualitative.pth')
    best_model_quantitative = load_model('./trained_models/best_model_quantitative.pth')

    print('===> Testing Wiz (qualitative)...')
    test_wiz(best_model_qualitative, output_folder='/project/exp/wiz_qual/')

    print('===> Testing Wiz (quantitative)...')
    test_wiz(best_model_quantitative, output_folder='/project/exp/wiz_quant/')

if __name__ == '__main__':
    test_all()
