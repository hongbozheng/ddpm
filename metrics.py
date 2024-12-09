#!/usr/bin/env python3

from numpy import ndarray

from config import get_config
import logger
import numpy as np
import os
import random
import scipy as sp
from logger import timestamp
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def fid(im_0: ndarray, im_1: ndarray) -> float:
    mu_0 = im_0.mean(axis=0)
    sig_0 = np.cov(m=im_0, rowvar=False)
    mu_1 = im_1.mean(axis=0)
    sig_1 = np.cov(m=im_1, rowvar=False)
    fid = np.sum(a=(mu_0 - mu_1) ** 2.0) \
        + np.trace(a=sig_0 + sig_1 - 2.0 * sp.linalg.sqrtm(A= sig_0 @ sig_1).real)
    return fid.item()


def main() -> None:
    cfg = get_config(args=None)

    class_names = []
    cls_fids = []
    cls_mssims = []

    img_names = os.listdir(path=cfg.IMAGE.DIR)

    img_tqdm = tqdm(iterable=img_names, position=0, leave=True)
    img_tqdm.set_description(
        desc=f"[{timestamp()}] {img_names[0]}",
        refresh=True,
    )

    for img_name in img_tqdm:
        img_tqdm.set_description(
            desc=f"[{timestamp()}] {img_name}",
            refresh=True,
        )
        img_path = os.path.join(cfg.IMAGE.DIR, img_name)
        im_0 = Image.open(img_path)
        im_0 = np.array(im_0)

        class_name = os.path.splitext(img_name)[0]
        class_names.append(class_name)

        fids = []
        mssims = []

        class_dir = os.path.join(cfg.DATA.DATA_DIR, class_name)
        filenames = os.listdir(class_dir)
        filenames = random.sample(filenames, k=int(0.1 * len(filenames)))

        file_tqdm = tqdm(iterable=filenames, position=1, leave=False)
        file_tqdm.set_description(
            desc=f"[{timestamp()}] {filenames[0]}",
            refresh=True,
        )

        for filename in file_tqdm:
            file_tqdm.set_description(
                desc=f"[{timestamp()}] {filename}",
                refresh=True,
            )
            filepath = os.path.join(class_dir, filename)
            im_1 = Image.open(filepath)
            im_1 = np.array(im_1)
            f = fid(im_0=im_0, im_1=im_1)
            fids.append(f)

            data_range = im_0.max() - im_0.min()

            mssis, S = ssim(
                im1=im_0,
                im2=im_1,
                win_size=7,
                data_range=data_range,
                channel_axis=None,
                full=True,
            )
            mssims.append(mssis)

        fid_mean = np.mean(a=fids, dtype=np.float64)
        cls_fids.append(fid_mean)
        cls_mssim_mean = np.mean(a=mssims, dtype=np.float64)
        cls_mssims.append(cls_mssim_mean)

    for class_name, f, s in zip(class_names, cls_fids, cls_mssims):
        logger.log_info(f"{class_name:<9}: FID {f:.6f} SSIM {s:.6f}")

    return


if __name__ == "__main__":
    main()
