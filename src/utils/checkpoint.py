import torch
import os
import logging

def save_checkpoint(state, work_dir, filename="latest.pth", is_best=False):
    os.makedirs(work_dir, exist_ok=True)
    filepath = os.path.join(work_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(work_dir, "best_model.pth")
        torch.save(state, best_path)

def load_checkpoint(model, filename, map_location='cpu'):
    logger = logging.getLogger(__name__)
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No checkpoint found at {filename}")

    logger.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location=map_location)

    state_dict = checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"Missing keys: {msg.missing_keys}")
    return checkpoint
