#!/usr/bin/env python3

from config import get_config, DEVICE
import copy
import torch.nn as nn
from dataset import MedicalMNIST
from ddpm import Diffusion
from ema import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from train import train_model
from unet import UNet


def main() -> None:
    cfg = get_config(args=None)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = MedicalMNIST(path=cfg.DATA.DATA_DIR, transform=transform)

    model = UNet(
        in_channels=cfg.MODEL.UNET.IN_CHANNELS,
        out_channels=cfg.MODEL.UNET.OUT_CHANNELS,
        n_channels=cfg.MODEL.UNET.N_CHANNELS,
        n_blks=cfg.MODEL.UNET.N_BLKS,
        attn=cfg.MODEL.UNET.ATTN,
        n_groups=cfg.MODEL.UNET.N_GROUPS,
        eps=cfg.MODEL.UNET.EPS,
        n_heads=cfg.MODEL.UNET.N_HEADS,
        t_emb_dim=cfg.MODEL.UNET.T_EMB_DIM,
        n_classes=len(train_dataset.cls2idx),
    )

    # param_size = 0
    # for param in model.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size) / 1024**2
    # print('model size: {:.3f}MB'.format(size_all_mb))

    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(pytorch_total_params)

    # return

    diffusion = Diffusion(
        noise_steps=cfg.MODEL.DDPM.NOISE_STEPS,
        beta_start=cfg.MODEL.DDPM.BETA_START,
        beta_end=cfg.MODEL.DDPM.BETA_END,
        img_size=cfg.MODEL.DDPM.IMG_SIZE,
        device=DEVICE,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.LOADER.TRAIN.BATCH_SIZE,
        shuffle=cfg.LOADER.TRAIN.SHUFFLE,
        num_workers=cfg.LOADER.TRAIN.NUM_WORKERS,
        pin_memory=cfg.LOADER.TRAIN.PIN_MEMORY,
    )
    # for batch in train_loader:
    #     print(batch["image"], batch["image"].shape)
    #     print(batch["label"], batch["label"].shape)
    #     break

    optimizer = AdamW(
        params=model.parameters(),
        lr=cfg.OPTIM.ADAMW.LR,
        weight_decay=cfg.OPTIM.ADAMW.WEIGHT_DECAY,
    )

    # lr_scheduler = CosineAnnealingLR(
    #     optimizer=optimizer,
    #     T_max=10,
    #     eta_min=1e-8,
    #     last_epoch=-1,
    # )

    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=cfg.LRS.CAWR.T_0,
        T_mult=cfg.LRS.CAWR.T_MULT,
        eta_min=cfg.LRS.CAWR.ETA_MIN,
        last_epoch=cfg.LRS.CAWR.LAST_EPOCH,
    )

    ema = EMA(beta=cfg.EMA.BETA)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    train_model(
        model=model,
        diffusion=diffusion,
        device=DEVICE,
        ckpt_filepath=cfg.BEST_MODEL.UNET,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        n_epochs=cfg.TRAIN.N_EPOCHS,
        train_loader=train_loader,
        criterion=nn.MSELoss(),
        ema=ema,
        ema_model=ema_model,
    )

    return


if __name__ == '__main__':
    main()
