"""
Training loop for CLAM / TumorAwareMIL.
Owner: Luke Zhao

TODO: implement Trainer.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.use_wandb = use_wandb

    def train_epoch(self, loader: DataLoader) -> float:
        raise NotImplementedError("Trainer.train_epoch to be implemented by Luke Zhao.")

    def evaluate(self, loader: DataLoader) -> dict:
        raise NotImplementedError("Trainer.evaluate to be implemented by Luke Zhao.")
