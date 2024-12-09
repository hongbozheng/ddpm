#!/usr/bin/env python3

from torch import Tensor
from typing import Dict

from config import get_config, DEVICE
import logger
import os
import torch
from dataset import MedicalMNIST
from ddpm import Diffusion
from PIL import Image
from torchvision.transforms import transforms
from unet import UNet


def save_images(images: Tensor, idx2cls: Dict, path: str) -> None:
    if not os.path.exists(path=path):
        os.makedirs(name=path, exist_ok=True)

    for i, image in enumerate(images):
        image = image.to('cpu').numpy().squeeze(axis=0)
        image = Image.fromarray(obj=image, mode='L')
        image.save(fp=f"{path}/{idx2cls[i]}.jpeg", format="JPEG")

    logger.log_info(f"Saved images to '{path}'.")

    return


def main() -> None:
    cfg = get_config(args=None)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = MedicalMNIST(path=cfg.DATA.DATA_DIR, transform=transform)
    n_classes = len(train_dataset.cls2idx)

    model = UNet(
        in_channels=cfg.MODEL.UNET.IN_CHANNELS,
        out_channels=cfg.MODEL.UNET.OUT_CHANNELS,
        channels=cfg.MODEL.UNET.CHANNELS,
        n_groups=cfg.MODEL.UNET.N_GROUPS,
        dropout=cfg.MODEL.UNET.DROPOUT,
        n_layers=cfg.MODEL.UNET.N_LAYERS,
        n_heads=cfg.MODEL.UNET.N_HEADS,
        n_classes=n_classes,    
        t_emb_dim=cfg.MODEL.UNET.T_EMB_DIM,
    )

    diffusion = Diffusion(
        noise_steps=cfg.MODEL.DDPM.NOISE_STEPS,
        beta_start=cfg.MODEL.DDPM.BETA_START,
        beta_end=cfg.MODEL.DDPM.BETA_END,
        img_size=cfg.MODEL.DDPM.IMG_SIZE,
        device=DEVICE,
    )

    labels = torch.arange(n_classes, dtype=torch.int64).to(device=DEVICE)

    images = diffusion.sample(
        model=model,
        device=DEVICE,
        ckpt_filepath=cfg.BEST_MODEL.UNET,
        labels=labels,
        cfg_scale=3,
    )

    idx2cls = train_dataset.idx2cls
    save_images(images=images, idx2cls=idx2cls, path=cfg.IMAGE.DIR)

    return


if __name__ == "__main__":
    main()  
