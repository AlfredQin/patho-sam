import numpy as np


def _pad_image(image, target_shape=(512, 512), pad_value=0):
    pad = [
        (max(t - s, 0) // 2, max(t - s, 0) - max(t - s, 0) // 2) for s, t in zip(image.shape[:2], target_shape)
    ]

    if image.ndim == 3:
        pad.append((0, 0))

    return np.pad(image, pad, mode="constant", constant_values=pad_value)