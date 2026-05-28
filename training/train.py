"""
Training loop for CLAM / TumorAwareMIL.
Owner: Luke Zhao
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import binary_metrics, multiclass_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        target: str,
        task_type: str = "binary",        # "binary" | "multiclass"
        device: str = "cuda",
        use_wandb: bool = False,
        instance_eval: bool = False,
        instance_loss_weight: float = 0.3,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.target = target
        self.task_type = task_type
        self.device = device
        self.use_wandb = use_wandb
        self.instance_eval = instance_eval
        self.instance_loss_weight = instance_loss_weight
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for features, labels in loader:
            # features: (1, K, D) with batch_size=1 → squeeze to (K, D)
            features = features.squeeze(0).to(self.device)
            target_label = labels[self.target].to(self.device)  # (1,)

            self.optimizer.zero_grad()

            if self.instance_eval:
                out = self.model(features, label=target_label, instance_eval=True)
                logits = out[0]
                instance_loss = out[2]  # always index 2: (logits, A, inst_loss[, tumor_probs])
                slide_loss = self.criterion(logits, target_label)
                loss = (1 - self.instance_loss_weight) * slide_loss + \
                       self.instance_loss_weight * instance_loss
            else:
                out = self.model(features)
                logits = out[0]
                loss = self.criterion(logits, target_label)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if self.use_wandb:
                import wandb
                wandb.log({"train/loss_step": loss.item()})

        return total_loss / max(len(loader), 1)

    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for features, labels in loader:
                features = features.squeeze(0).to(self.device)
                label = int(labels[self.target].item())

                out = self.model(features)
                logits = out[0]                                     # (1, n_classes)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # (n_classes,)

                all_probs.append(probs)
                all_labels.append(label)

        all_labels = np.array(all_labels)
        all_probs = np.stack(all_probs)  # (N, n_classes)

        if self.task_type == "binary":
            metrics = binary_metrics(all_labels, all_probs[:, 1])
        else:
            metrics = multiclass_metrics(all_labels, all_probs)

        if self.use_wandb:
            import wandb
            wandb.log({f"val/{k}": v for k, v in metrics.items()})

        return metrics
