from torch import Tensor
from typing import Tuple

import logger
import os
import torch
import torch.nn as nn
from logger import timestamp
from tqdm import tqdm


class Diffusion:
    def __init__(
            self,
            noise_steps: int,
            beta_start: float,
            beta_end: float,
            img_size: int,
            device: torch.device,
    ) -> None:
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.noise_schedule().to(device=device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(input=self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

        return

    def noise_schedule(self) -> Tensor:
        return torch.linspace(
            start=self.beta_start,
            end=self.beta_end,
            steps=self.noise_steps,
        )

    def noise_image(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        # [N] -> [N, 1, 1, 1]
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
        # [N] -> [N, 1, 1, 1]
        sqrt_1_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).view(-1, 1, 1, 1)
        eps = torch.randn_like(input=x)

        return sqrt_alpha_hat * x + sqrt_1_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n: int) -> Tensor:
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(
            self,
            model: nn.Module,
            device: torch.device,
            ckpt_filepath: str,
            labels: Tensor,
            cfg_scale: int,
    ) -> Tensor:
        model.to(device=device)
        model.eval()

        if os.path.exists(path=ckpt_filepath):
            ckpt = torch.load(f=ckpt_filepath, map_location=device)
            model.load_state_dict(state_dict=ckpt["model"])
            filename = os.path.basename(p=ckpt_filepath)
            logger.log_info(f"Loaded '{filename}'")

        sample_tqdm = tqdm(
            iterable=range(self.noise_steps - 1, 0, -1),
            position=0,
            leave=False,
        )
        sample_tqdm.set_description(
            desc=f"[{timestamp()}] [Sample {self.noise_steps}]",
            refresh=True,
        )

        n = labels.size(dim=0)

        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)

            for i in sample_tqdm:
                t = torch.full((n,), i, dtype=torch.int64, device=device)
                pred_noise = model(x=x, t=t, y=labels)
                if cfg_scale > 0:
                    uncond_pred_noise = model(x=x, t=t, y=None)
                    pred_noise = torch.lerp(
                        input=uncond_pred_noise,
                        end=pred_noise,
                        weight=cfg_scale,
                    )
                alpha = self.alpha[t].view(-1, 1, 1, 1)
                alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
                beta = self.beta[t].view(-1, 1, 1, 1)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (
                    torch.sqrt(1 - alpha_hat))) * pred_noise)+ torch.sqrt(beta) * noise

                sample_tqdm.set_description(
                    desc=f"[{timestamp()}] [Sample {i}]",
                    refresh=True,
                )

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).to(dtype=torch.uint8)

        return x
