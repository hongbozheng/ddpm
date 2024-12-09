from numpy import ndarray
from typing import Tuple, Union

from config import get_config
import numpy as np
import os
import scipy as sp
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def fid(real: ndarray, generate: ndarray) -> ndarray:
    mu_0 = real.mean(axis=0)
    sig_0 = np.cov(m=real, rowvar=False)
    mu_1 = generate.mean(axis=0)
    sig_1 = np.cov(m=generate, rowvar=False)
    fid = np.sum(a=(mu_0 - mu_1) ** 2.0) \
        + np.trace(a=sig_0 + sig_1 - 2.0 * sp.sqrtm(A= sig_0 @ sig_1))
    return fid


def main() -> None:
    cfg = get_config(args=None)

    img_paths = os.listdir(path=cfg.IMAGE.DIR)

    fid = []
    mssim = []

    for img_path in img_paths:
        image = Image.open(img_path)
        image = np.array(image)

        classes = os.listdir(path=cfg.DATA.DATA_DIR)

        filename = os.path.splitext(os.path.basename(img_path))[0]

        fids = []
        mssims = []

        filepath = os.path.join(cfg.DATA.DATA_DIR, filename)
        filepaths = os.listdir(filepath)
        for filepath in filepaths:
            im1 = Image.open(img_path)
            im1 = np.array(im1)

            fid(real=image, generate=im1)
            fids.append(fid)
    
        fid.append(np.array(fids).mean())

    # mssis, S = ssim(
    #     im1=im_0,
    #     im2=im_1,
    #     win_size=win_size,
    #     data_range=data_range,
    #     channel_axis=channel_axis,
    #     full=full,
    # )

    return


if __name__ == "__main__":
    main()
