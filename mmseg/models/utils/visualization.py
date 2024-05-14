# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add OEM and FLAIR palette

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from mmseg.models.utils.dacs_transforms import denorm


OEM_PALETTE = [0, 0, 0, 128, 0, 0, 0, 255, 36, 148, 148, 148, 255, 255, 255,
               34, 97, 38, 0, 69, 255, 75, 181, 73, 222, 31, 7]

FLAIR_PALETTE = [241, 91, 181, 229, 228, 233, 157, 2, 8, 55, 7, 23, 68, 96, 239,
                18, 42, 18, 81, 183, 136, 232, 92, 4, 113, 9, 182, 79, 119, 46,
                255, 242, 13, 244, 140, 5, 0, 0, 0]


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def _colorize(img, cmap, mask_zero=False):
    vmin = np.min(img)
    vmax = np.max(img)
    mask = (img <= 0).squeeze()
    cm = plt.get_cmap(cmap)
    colored_image = cm(np.clip(img.squeeze(), vmin, vmax) / vmax)[:, :, :3]
    # Use white if no depth is available (<= 0)
    if mask_zero:
        colored_image[mask, :] = [1, 1, 1]
    return colored_image


def get_segmentation_error_vis(seg, gt):
    error_mask = seg != gt
    error_mask[gt == 255] = 0
    out = seg.copy()
    out[error_mask == 0] = 255
    return out


def is_integer_array(a):
    return np.all(np.equal(np.mod(a, 1), 0))


def prepare_debug_out(title, out, mean, std, cmap):
    if len(out.shape) == 4 and out.shape[0] == 1:
        out = out[0]
    if len(out.shape) == 2:
        out = np.expand_dims(out, 0)
    assert len(out.shape) == 3
    if out.shape[0] == 3:
        if mean is not None:
            out = torch.clamp(denorm(out, mean, std), 0, 1)[0]
        out = dict(title=title, img=out)
    elif out.shape[0] > 3:
        out = torch.softmax(torch.from_numpy(out), dim=0).numpy()
        out = np.argmax(out, axis=0)
        out = dict(title=title, img=out, cmap=cmap)
    elif out.shape[0] == 1:
        if is_integer_array(out) and np.max(out) > 1:
            out = dict(title=title, img=out[0], cmap=cmap)
        elif np.min(out) >= 0 and np.max(out) <= 1:
            out = dict(title=title, img=out[0], cmap=cmap, vmin=0, vmax=1)
        else:
            out = dict(title=title, img=out[0], cmap=cmap, range_in_title=True)
    else:
        raise NotImplementedError(out.shape)
    return out


def subplotimg(ax,
               img,
               title=None,
               range_in_title=False,
               **kwargs):
    if img is None:
        return
    with torch.no_grad():
        if torch.is_tensor(img):
            img = img.cpu()
        if len(img.shape) == 2:
            if torch.is_tensor(img):
                img = img.numpy()
        elif img.shape[0] == 1:
            if torch.is_tensor(img):
                img = img.numpy()
            img = img.squeeze(0)
        elif img.shape[0] == 3:
            img = img.permute(1, 2, 0)
            if not torch.is_tensor(img):
                img = img.numpy()
        
        if kwargs.get('cmap', '') == 'oem':
            kwargs.pop('cmap')
            if torch.is_tensor(img):
                img = img.numpy()
            img = colorize_mask(img, palette=OEM_PALETTE)
            
        if kwargs.get('cmap', '') == 'flair':
            kwargs.pop('cmap')
            if torch.is_tensor(img):
                img = img.numpy()
            img = colorize_mask(img, palette=FLAIR_PALETTE)

    if range_in_title:
        vmin = np.min(img)
        vmax = np.max(img)
        title += f' {vmin:.3f}-{vmax:.3f}'

    ax.imshow(img, **kwargs)
    if title is not None:
        ax.set_title(title)
