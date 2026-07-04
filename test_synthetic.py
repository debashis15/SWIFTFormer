## SWIFTFormer -- paired / synthetic benchmark testing
##
## Evaluates on paired datasets (LOL, LOLv2-Synthetic, LOLv2-Real,
## MIT-Adobe FiveK) with ground-truth references and reports
## PSNR / SSIM / LPIPS. Enhanced results are saved to --result_dir.
##
## Expected data layout:
##   <input_dir>/low/*.png     low-light inputs
##   <input_dir>/high/*.png    normal-light ground truth
##
## Usage:
##   python test_synthetic.py \
##       --input_dir ./datasets/LOL/test \
##       --result_dir ./results/LOL \
##       --weights ./checkpoints/SWIFTFormer_LOL.pth

import argparse
import os

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from datasets import get_validation_data
from model import SWIFTFormer


def main():
    parser = argparse.ArgumentParser(
        description='SWIFTFormer evaluation on paired (synthetic) benchmarks')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='dataset root containing low/ and high/ folders')
    parser.add_argument('--result_dir', type=str, default='./results/synthetic',
                        help='directory to save enhanced outputs')
    parser.add_argument('--weights', type=str, required=True,
                        help='path to pre-trained model weights')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--save_images', action='store_true', default=True)
    parser.add_argument('--no_lpips', action='store_true',
                        help='disable LPIPS computation')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    utils.mkdir(args.result_dir)

    # ------------------------- model -------------------------
    model = SWIFTFormer().to(device)
    utils.load_checkpoint(model, args.weights)
    model.eval()
    print(f"==> Loaded weights: {args.weights}")
    print(f"==> Parameters: {utils.network_parameters(model) / 1e6:.2f} M")

    # ------------------------ LPIPS --------------------------
    lpips_fn = None
    if not args.no_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex').to(device)
        except ImportError:
            print('[warn] lpips not installed -- skipping LPIPS '
                  '(pip install lpips)')

    # ------------------------- data --------------------------
    test_dataset = get_validation_data(args.input_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=4, pin_memory=True)

    psnr_all, ssim_all, lpips_all = [], [], []

    with torch.no_grad():
        for target, input_, filename in tqdm(test_loader, ncols=90):
            target = target.to(device)
            input_ = input_.to(device)

            restored = torch.clamp(model(input_), 0, 1)

            if lpips_fn is not None:
                lpips_all.append(
                    lpips_fn(restored * 2 - 1, target * 2 - 1).item())

            restored_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
            target_np = target.squeeze(0).permute(1, 2, 0).cpu().numpy()

            psnr_all.append(compare_psnr(target_np, restored_np, data_range=1.0))
            ssim_all.append(compare_ssim(target_np, restored_np,
                                         channel_axis=2, data_range=1.0))

            if args.save_images:
                out = (restored_np * 255.0).round().astype(np.uint8)
                utils.save_img(os.path.join(args.result_dir,
                                            filename[0] + '.png'), out)

    print('\n==================== Results ====================')
    print(f'Dataset : {args.input_dir}')
    print(f'PSNR    : {np.mean(psnr_all):.4f} dB')
    print(f'SSIM    : {np.mean(ssim_all):.4f}')
    if lpips_all:
        print(f'LPIPS   : {np.mean(lpips_all):.4f}')
    print('=================================================')


if __name__ == '__main__':
    main()
