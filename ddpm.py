import torch
from torch import Tensor

import logger

class Diffusion:
    def __init__(
            self,
            steps: int,
            beta_start: float,
            beta_end: float,
            img_size: int,
            device: torch.device
    ) -> None:
        return