import cv2
import torch
def load_mask_bool(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask file not found or unreadable: {mask_path}")

    if mask.ndim == 2:
        mask_alpha = mask
    elif mask.ndim == 3 and mask.shape[2] == 4:
        # Prefer alpha channel for RGBA-style masks.
        mask_alpha = mask[..., 3]
    elif mask.ndim == 3:
        # Regular PNG/JPG without alpha: treat any non-zero pixel as foreground.
        mask_alpha = mask.max(axis=2)
    else:
        raise ValueError(f"Unsupported mask shape {mask.shape} for file: {mask_path}")

    return mask_alpha > 0