# sft_loss.py
"""
Loss functions for Supervised Fine-Tuning (SFT).
Implements masked cross-entropy loss that ignores prompt tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SFTLoss(nn.Module):
    """
    Masked cross-entropy loss used for SFT.
    Supports:
    - ignore_index
    - attention_mask
    - label smoothing
    - accuracy calculation
    """

    def __init__(self, ignore_index=-100, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels, attention_mask=None, return_metrics=False):
        B, T, V = logits.shape

        logits_flat = logits.reshape(B*T, V)
        labels_flat = labels.reshape(B*T)

        # ------- valid mask -------
        valid_mask = (labels_flat != self.ignore_index)
        if attention_mask is not None:
            attention_flat = attention_mask.reshape(B*T).bool()
            valid_mask = valid_mask & attention_flat

        # ------- loss -------
        if self.label_smoothing > 0:
            loss_vec = F.cross_entropy(
                logits_flat,
                labels_flat,
                reduction="none",
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
            )
        else:
            loss_vec = F.cross_entropy(
                logits_flat,
                labels_flat,
                reduction="none",
                ignore_index=self.ignore_index,
            )

        loss_vec = loss_vec * valid_mask.float()

        if self.reduction == "mean":
            denom = valid_mask.sum().clamp(min=1)
            loss = loss_vec.sum() / denom
        elif self.reduction == "sum":
            loss = loss_vec.sum()
        else:
            loss = loss_vec.reshape(B, T)  # rarely used

        if not return_metrics:
            return loss

        # ------- accuracy -------
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # [B, T]
            correct = ((preds == labels) & valid_mask.reshape(B, T)).sum()
            total = valid_mask.sum()
            accuracy = (correct.float() / total.float()) if total > 0 else torch.tensor(0.0, device=logits.device)

            metrics = {
                "loss": float(loss.detach().cpu()),
                "accuracy": float(accuracy.detach().cpu()),
                "num_valid_tokens": int(total.cpu()),
            }

        return loss, metrics


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Convenience function to compute SFT loss.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len] 
        attention_mask: Optional mask [batch_size, seq_len]
        ignore_index: Token ID to ignore
    
    Returns:
        Loss tensor (scalar)
    """
    loss_fn = SFTLoss(ignore_index=ignore_index)
    return loss_fn(logits, labels, attention_mask)

