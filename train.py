## SWIFTFormer training script
##
## Implementation details (Sec. IV-B of the paper):
##   - Adam optimizer, beta1 = 0.9, beta2 = 0.999
##   - ~2.5e5 iterations
##   - lr: 2e-4 -> 1e-6 via cosine annealing
##   - 128x128 random crops, batch size 8
##   - augmentation: random rotation + horizontal flip
##   - Charbonnier loss
##
## Usage:
##   python train.py --yml_path configs/LOL/train/training_LOL.yaml

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from datasets import get_training_data, get_validation_data
from model import SWIFTFormer
from utils.losses import CharbonnierLoss


def main():
    parser = argparse.ArgumentParser(description='SWIFTFormer training')
    parser.add_argument('--yml_path', type=str,
                        default='configs/LOL/train/training_LOL.yaml',
                        help='path to the training config yaml')
    args = parser.parse_args()

    with open(args.yml_path, 'r') as f:
        opt = yaml.safe_load(f)

    train_opt = opt['TRAINING']
    optim_opt = opt['OPTIM']
    model_opt = opt.get('MODEL', {})

    # reproducibility
    seed = train_opt.get('SEED', 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------- model -------------------------
    model = SWIFTFormer(
        in_channels=model_opt.get('IN_CHANNELS', 3),
        out_channels=model_opt.get('OUT_CHANNELS', 3),
        dim=model_opt.get('DIM', 36),
        num_blocks=tuple(model_opt.get('NUM_BLOCKS', [2, 3, 3, 4])),
        num_refinement_blocks=model_opt.get('NUM_REFINEMENT_BLOCKS', 4),
        heads=tuple(model_opt.get('HEADS', [1, 2, 4, 8])),
        ffn_expansion_factor=model_opt.get('FFN_EXPANSION_FACTOR', 2.85),
        wave=model_opt.get('WAVE', 'db4'),
    ).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    print(f"==> SWIFTFormer parameters: {utils.network_parameters(model) / 1e6:.2f} M")

    # ----------------------- optimizer -----------------------
    start_epoch = 1
    lr_initial = float(optim_opt['LR_INITIAL'])
    lr_min = float(optim_opt['LR_MIN'])
    num_epochs = int(optim_opt['NUM_EPOCHS'])

    optimizer = optim.Adam(model.parameters(), lr=lr_initial,
                           betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr_min)

    # ------------------------ resume -------------------------
    model_dir = os.path.join(train_opt['SAVE_DIR'], 'models')
    utils.mkdir(model_dir)

    if train_opt.get('RESUME', False):
        path_chk = utils.dir_utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model, path_chk)
        start_epoch = utils.load_start_epoch(path_chk) + 1
        utils.load_optim(optimizer, path_chk)
        for _ in range(1, start_epoch):
            scheduler.step()
        print(f"==> Resuming from epoch {start_epoch}, lr = {scheduler.get_last_lr()[0]:.2e}")

    # ------------------------- data --------------------------
    train_dataset = get_training_data(
        train_opt['TRAIN_DIR'], {'patch_size': train_opt['TRAIN_PS']})
    train_loader = DataLoader(train_dataset,
                              batch_size=optim_opt['BATCH_SIZE'],
                              shuffle=True,
                              num_workers=train_opt.get('NUM_WORKERS', 8),
                              drop_last=True,
                              pin_memory=True)

    val_dataset = get_validation_data(train_opt['VAL_DIR'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)

    criterion = CharbonnierLoss().to(device)

    # ------------------------- train -------------------------
    best_psnr, best_epoch = 0.0, 0
    print(f"==> Training from epoch {start_epoch} to {num_epochs}")

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for data in tqdm(train_loader, desc=f'Epoch {epoch}', ncols=90):
            target = data[0].to(device, non_blocking=True)
            input_ = data[1].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            restored = model(input_)
            loss = criterion(restored, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # ---------------------- validation ----------------------
        if epoch % train_opt.get('VAL_AFTER_EVERY', 1) == 0:
            model.eval()
            psnr_val = []
            with torch.no_grad():
                for data_val in val_loader:
                    target = data_val[0].to(device)
                    input_ = data_val[1].to(device)
                    restored = torch.clamp(model(input_), 0, 1)
                    psnr_val.append(utils.torchPSNR(target, restored).item())
            psnr_val = float(np.mean(psnr_val))

            if psnr_val > best_psnr:
                best_psnr, best_epoch = psnr_val, epoch
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join(model_dir, 'model_best.pth'))

            print(f"[epoch {epoch}] PSNR: {psnr_val:.4f} dB "
                  f"(best: {best_psnr:.4f} dB @ epoch {best_epoch})")

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(model_dir, 'model_latest.pth'))

        print(f"Epoch {epoch} | time: {time.time() - epoch_start:.1f}s | "
              f"loss: {epoch_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

    print(f"==> Done. Best PSNR {best_psnr:.4f} dB at epoch {best_epoch}.")


if __name__ == '__main__':
    main()
