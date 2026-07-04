import os
from collections import OrderedDict

import torch


def network_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def save_checkpoint(model_dir, state, session=''):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir, f'model_epoch_{epoch}_{session}.pth')
    torch.save(state, model_out_path)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except Exception:
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_start_epoch(weights):
    checkpoint = torch.load(weights, map_location='cpu')
    return checkpoint['epoch']


def load_optim(optimizer, weights):
    checkpoint = torch.load(weights, map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups:
        lr = p['lr']
    return lr
