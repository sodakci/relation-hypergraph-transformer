"""支持未标注掩码的 AR/WW 双头训练损失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import DecisionOutputs


@dataclass(frozen=True)
class DecisionLoss:
    """总损失、分头损失以及实际参与训练的样本数。"""

    total: Tensor
    ar: Tensor
    ww: Tensor
    ar_examples: int
    ww_examples: int


def _head_loss(
    logits: Tensor,
    indices: Tensor,
    labels: Tensor,
    labeled_mask: Tensor,
    pos_weight: Tensor | None,
) -> tuple[Tensor, int]:
    """从一个输出头中选择有标签样本并计算 BCEWithLogitsLoss。"""

    # indices 将该头的紧凑 logits 映射回原始 Decision 编号。
    selected = labeled_mask[indices]
    selected_logits = logits[selected]
    selected_labels = labels[indices][selected].to(dtype=logits.dtype)
    if selected_logits.numel() == 0:
        # 返回连接在计算图上的零值，保证某类样本缺失时仍可安全 backward。
        return logits.sum() * 0.0, 0
    return (
        F.binary_cross_entropy_with_logits(
            selected_logits,
            selected_labels,
            pos_weight=pos_weight,
        ),
        selected_logits.numel(),
    )


def decision_loss(
    outputs: DecisionOutputs,
    labels: Tensor,
    *,
    labeled_mask: Tensor | None = None,
    ar_pos_weight: Tensor | None = None,
    ww_pos_weight: Tensor | None = None,
    ww_weight: float = 1.0,
) -> DecisionLoss:
    """仅对有精确标签的 Decision 计算损失，并分别统计 AR 与 WW。"""

    if labels.ndim != 1 or not labels.is_floating_point():
        raise TypeError("labels must be a one-dimensional floating-point tensor")
    decision_count = outputs.ar_indices.numel() + outputs.ww_indices.numel()
    if labels.numel() != decision_count:
        raise ValueError(f"labels must contain one value per decision ({decision_count})")
    if labels.device != outputs.ar_logits.device or labels.device != outputs.ww_logits.device:
        raise ValueError("labels and logits must be on the same device")
    # 未提供掩码表示所有 Decision 都拥有可靠标签。
    if labeled_mask is None:
        labeled_mask = torch.ones_like(labels, dtype=torch.bool)
    if labeled_mask.shape != labels.shape or labeled_mask.dtype != torch.bool:
        raise TypeError("labeled_mask must be a bool tensor with the same shape as labels")
    if labeled_mask.device != labels.device:
        raise ValueError("labeled_mask and labels must be on the same device")
    if ww_weight < 0:
        raise ValueError("ww_weight cannot be negative")
    if labels.numel() and torch.any((labels[labeled_mask] != 0) & (labels[labeled_mask] != 1)):
        raise ValueError("labeled targets must be binary (0 or 1)")

    # pos_weight 分别处理 AR/WW 的类不平衡；ww_weight 控制两项任务的相对权重。
    ar_loss, ar_examples = _head_loss(
        outputs.ar_logits,
        outputs.ar_indices,
        labels,
        labeled_mask,
        ar_pos_weight,
    )
    ww_loss, ww_examples = _head_loss(
        outputs.ww_logits,
        outputs.ww_indices,
        labels,
        labeled_mask,
        ww_pos_weight,
    )
    return DecisionLoss(
        total=ar_loss + ww_weight * ww_loss,
        ar=ar_loss,
        ww=ww_loss,
        ar_examples=ar_examples,
        ww_examples=ww_examples,
    )
