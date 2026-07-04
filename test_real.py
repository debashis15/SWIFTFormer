## SWIFTFormer -- real-world / unpaired benchmark testing
##
## Enhances unpaired real-world low-light datasets (DICM, LIME, MEF,
## NPE) that have no ground-truth references, and reports the
## no-reference NIQE metric (lower is better). Enhanced results are
## saved to --result_dir.
##
## Expected data layout:
##   <input_dir>/*.png|jpg|bmp    low-light inputs only (no GT)
##
## Usage:
##   python test_real.py \
##       --input_dir ./datasets/DICM \
##       --result_dir ./results/DICM \
##       --weights ./checkpoints/SWIFTFormer_LOL.pth

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from datasets import get_test_unpaired_data
from model import SWIFTFormer


def main():
    parser = argparse.ArgumentParser(
        description='SWIFTFormer evaluation on unpaired real-world benchmarks')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='directory of unpaired low-light images')
    parser.add_argument('--result_dir', type=str, default='./results/real',
                        help='directory to save enhanced outputs')
    parser.add_argument('--weights', type=str, required=True,
                        help='path to pre-trained model weights')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--no_niqe', action='store_true',
                        help='disable NIQE computation')
    parser.add_argument('--max_size', type=int, default=0,
                        help='optional: downscale longest side to this size '
                             'before inference (0 = full resolution)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    utils.mkdir(args.result_dir)

    # ------------------------- model -------------------------
    model = SWIFTFormer().to(device)
    utils.load_checkpoint(model, args.weights)
    model.eval()
    print(f"==> Loaded weights: {args.weights}")

    # ------------------------- NIQE --------------------------
    niqe_fn = None
    if not args.no_niqe:
        try:
            import pyiqa
            niqe_fn = pyiqa.create_metric('niqe', device=device)
        except ImportError:
            print('[warn] pyiqa not installed -- skipping NIQE '
                  '(pip install pyiqa)')

    # ------------------------- data --------------------------
    test_dataset = get_test_unpaired_data(args.input_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=4, pin_memory=True)

    niqe_all = []

    with torch.no_grad():
        for input_, filename in tqdm(test_loader, ncols=90):
            input_ = input_.to(device)

            # optional resize for very large real-world images
            _, _, h, w = input_.shape
            if args.max_size > 0 and max(h, w) > args.max_size:
                scale = args.max_size / max(h, w)
                input_ = F.interpolate(input_,
                                       scale_factor=scale,
                                       mode='bilinear',
                                       align_corners=False)

            restored = torch.clamp(model(input_), 0, 1)

            if niqe_fn is not None:
                niqe_all.append(niqe_fn(restored).item())

            restored_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out = (restored_np * 255.0).round().astype(np.uint8)
            utils.save_img(os.path.join(args.result_dir,
                                        filename[0] + '.png'), out)

    print('\n==================== Results ====================')
    print(f'Dataset : {args.input_dir}')
    print(f'Images  : {len(test_dataset)} enhanced -> {args.result_dir}')
    if niqe_all:
        print(f'NIQE    : {np.mean(niqe_all):.4f} '
              f'(+/- {np.std(niqe_all):.4f})  [lower is better]')
    print('=================================================')


if __name__ == '__main__':
    main()
