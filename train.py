import logger
import os
import torch
import torch.nn as nn
import torch.optim as optim
from avg_meter import AverageMeter
from ddpm import Diffusion
from ema import EMA
from logger import timestamp
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
        model: nn.Module,
        diffusion: Diffusion,
        train_loader: DataLoader,
        device: torch.device,
        optimizer: optim.Optimizer,
        criterion: nn.MSELoss,
        ema: EMA,
        ema_model: nn.Module,
) -> float:
    model.train(mode=True)

    loader_tqdm = tqdm(iterable=train_loader, position=1, leave=False)
    loader_tqdm.set_description(desc=f"[{timestamp()}] [Batch 0]", refresh=True)

    loss_meter = AverageMeter()

    for i, batch in enumerate(loader_tqdm):
        image = batch["image"].to(device=device)
        label = batch["label"].to(device=device)
        t = diffusion.sample_timesteps(n=image.size(dim=0)).to(device=device)
        x_t, noise = diffusion.noise_image(x=image, t=t)
        if torch.rand(1) < 0.1:
            label = None
        pred_noise = model(x=x_t, t=t, y=label)
        loss = criterion(noise, pred_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.step_ema(ema_model=ema_model, model=model)

        loss_meter.update(loss.item(), n=image.size(dim=0))

        loader_tqdm.set_description(
            desc=f"[{timestamp()}] [Batch {i + 1}]: train loss {loss_meter.avg:.6f}",
            refresh=True,
        )

        # sampled_images = diffusion.sample(model, n=image.shape[0])
        # save_images(sampled_images,
        #             os.path.join("results", args.run_name, f"{epoch}.jpg"))
        # torch.save(model.state_dict(),
        #            os.path.join("models", args.run_name, f"ckpt.pt"))

    return loss_meter.avg


def train_model(
        model: nn.Module,
        diffusion: Diffusion,
        device: torch.device,
        ckpt_filepath: str,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
        n_epochs: int,
        train_loader: DataLoader,
        criterion: nn.MSELoss,
        ema: EMA,
        ema_model: nn.Module,
) -> None:
    model.to(device=device)
    model.train(mode=True)
    ema_model.to(device=device)

    path, _ = os.path.split(p=ckpt_filepath)
    if not os.path.exists(path=path):
        os.makedirs(name=path, exist_ok=True)

    init_epoch = 0
    best_loss = float('inf')

    if os.path.exists(path="models/last_unet.ckpt"):
        ckpt = torch.load(f="models/last_unet.ckpt", map_location=device)
        model.load_state_dict(state_dict=ckpt["model"])
        optimizer.load_state_dict(state_dict=ckpt["optimizer"])
        lr_scheduler.load_state_dict(state_dict=ckpt["lr_scheduler"])
        init_epoch = ckpt["epoch"]+1
        best_loss = ckpt["best_loss"]
        state_dict = torch.load(f="models/last_ema.pt", map_location=device)
        ema_model.load_state_dict(state_dict=state_dict)
        filename = os.path.basename(p=ckpt_filepath)
        logger.log_info(f"Loaded '{filename}'")

    epoch_tqdm = tqdm(
        iterable=range(init_epoch, n_epochs),
        desc=f"[{timestamp()}] [Epoch {init_epoch}]",
        position=0,
        leave=True,
    )

    for epoch in epoch_tqdm:
        epoch_tqdm.set_description(
            desc=f"[{timestamp()}] [Epoch {epoch}]",
            refresh=True,
        )
        avg_loss = train_epoch(
            model=model,
            diffusion=diffusion,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            ema=ema,
            ema_model=ema_model,
        )
        lr_scheduler.step()

        epoch_tqdm.write(s=f"[{timestamp()}] [Epoch {epoch}]: loss {avg_loss:.6f}")

        if avg_loss <  best_loss:
            best_loss = avg_loss
            torch.save(
                obj={
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                },
                f=ckpt_filepath,
            )
            torch.save(obj=ema_model.state_dict(), f="models/ema.pt")
            epoch_tqdm.write(
                s=f"[{timestamp()}] [Epoch {epoch}]: Saved best model to "
                  f"'{ckpt_filepath}'"
            )

        torch.save(
                obj={
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                },
                f="models/last_unet.ckpt",
            )
        torch.save(obj=ema_model.state_dict(), f="models/last_ema.pt")
        epoch_tqdm.write(
            s=f"[{timestamp()}] [Epoch {epoch}]: Saved last model to "
               "'models/last_unet.ckpt'"
        )

    return
