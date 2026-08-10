"""VeriStrong WW 二分类和 WR 可行集合分组损失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .veristrong_model import VeriStrongOutputs
from .veristrong_graph import VeriStrongGraph


@dataclass(frozen=True)
class VeriStrongTargets:
    """支持单解、backbone 和 WR 多可行候选的监督标签。"""

    ww_labels: Tensor
    ww_labeled_mask: Tensor
    wr_feasible_mask: Tensor
    wr_labeled_group_mask: Tensor

    def to(self, device: torch.device | str) -> "VeriStrongTargets":
        return VeriStrongTargets(
            ww_labels=self.ww_labels.to(device),
            ww_labeled_mask=self.ww_labeled_mask.to(device),
            wr_feasible_mask=self.wr_feasible_mask.to(device),
            wr_labeled_group_mask=self.wr_labeled_group_mask.to(device),
        )


@dataclass(frozen=True)
class VeriStrongLoss:
    total: Tensor
    ww: Tensor
    wr: Tensor
    ww_examples: int
    wr_groups: int


def collate_veristrong_targets(targets: list[VeriStrongTargets]) -> VeriStrongTargets:
    """按与 ``collate_veristrong_graphs`` 相同的图顺序拼接监督张量。"""

    if not targets:
        raise ValueError("至少需要一组标签才能构造 batch")
    return VeriStrongTargets(
        ww_labels=torch.cat([target.ww_labels for target in targets]),
        ww_labeled_mask=torch.cat([target.ww_labeled_mask for target in targets]),
        wr_feasible_mask=torch.cat([target.wr_feasible_mask for target in targets]),
        wr_labeled_group_mask=torch.cat(
            [target.wr_labeled_group_mask for target in targets]
        ),
    )


def build_veristrong_targets(
    graph: VeriStrongGraph,
    *,
    ww_labels: dict[int, bool],
    wr_feasible_writers: dict[int, set[int]],
) -> VeriStrongTargets:
    """把使用原始 group/transaction ID 的标签映射到当前图的紧凑顺序。"""

    if graph.graph_count != 1:
        raise ValueError("原始 ID 标签只能直接映射到单图")
    ww_values = torch.zeros(len(graph.ww_choice_ids), dtype=torch.float32)
    ww_labeled = torch.zeros(len(graph.ww_choice_ids), dtype=torch.bool)
    for index, group_id in enumerate(graph.ww_choice_ids):
        if group_id in ww_labels:
            ww_values[index] = float(ww_labels[group_id])
            ww_labeled[index] = True

    wr_feasible = torch.zeros(int(graph.wr_mask.sum().item()), dtype=torch.bool)
    wr_labeled_groups = torch.zeros(graph.wr_group_count, dtype=torch.bool)
    wr_decisions = torch.nonzero(graph.wr_mask, as_tuple=False).squeeze(-1)
    for compact_index, decision_index in enumerate(wr_decisions.tolist()):
        dense_group = int(graph.decision_wr_group[decision_index].item())
        raw_group = graph.wr_choice_ids[dense_group]
        if raw_group not in wr_feasible_writers:
            continue
        wr_labeled_groups[dense_group] = True
        writer_local = int(graph.decision_left_transaction[decision_index].item())
        writer_id = graph.transaction_ids[writer_local]
        wr_feasible[compact_index] = writer_id in wr_feasible_writers[raw_group]
    for dense_group in torch.nonzero(wr_labeled_groups, as_tuple=False).squeeze(-1).tolist():
        members = graph.decision_wr_group[wr_decisions] == dense_group
        if not torch.any(wr_feasible[members]):
            raw_group = graph.wr_choice_ids[dense_group]
            raise ValueError(f"WR group {raw_group} 的可行 writer 不在当前候选中")

    return VeriStrongTargets(
        ww_labels=ww_values,
        ww_labeled_mask=ww_labeled,
        wr_feasible_mask=wr_feasible,
        wr_labeled_group_mask=wr_labeled_groups,
    )


def _segment_logsumexp(values: Tensor, groups: Tensor, group_count: int) -> Tensor:
    """仅使用原生 PyTorch 计算可变长度分组的 logsumexp。"""

    if values.ndim != 1 or groups.shape != values.shape or groups.dtype != torch.long:
        raise ValueError("values/groups 必须是等长一维张量，groups 使用 torch.long")
    if group_count == 0:
        return values.new_empty((0,))
    maxima = values.new_full((group_count,), -torch.inf)
    maxima.scatter_reduce_(0, groups, values, reduce="amax", include_self=True)
    sums = values.new_zeros((group_count,))
    sums.scatter_add_(0, groups, torch.exp(values - maxima[groups]))
    return maxima + torch.log(sums)


def veristrong_decision_loss(
    outputs: VeriStrongOutputs,
    targets: VeriStrongTargets,
    *,
    ww_pos_weight: Tensor | None = None,
    wr_weight: float = 1.0,
) -> VeriStrongLoss:
    """计算 WW BCE 与 WR feasible-set grouped cross-entropy。"""

    if wr_weight < 0:
        raise ValueError("wr_weight 不能为负数")
    if targets.ww_labels.shape != outputs.ww_logits.shape:
        raise ValueError("ww_labels 与 ww_logits 形状不一致")
    if targets.ww_labeled_mask.shape != outputs.ww_logits.shape or targets.ww_labeled_mask.dtype != torch.bool:
        raise ValueError("ww_labeled_mask 必须是与 ww_logits 等长的 bool 张量")
    if targets.wr_feasible_mask.shape != outputs.wr_logits.shape or targets.wr_feasible_mask.dtype != torch.bool:
        raise ValueError("wr_feasible_mask 必须是与 wr_logits 等长的 bool 张量")
    if targets.ww_labels.device != outputs.ww_logits.device:
        raise ValueError("WW 标签和 logits 必须位于同一设备")
    if targets.ww_labeled_mask.device != outputs.ww_logits.device:
        raise ValueError("WW 标签掩码和 logits 必须位于同一设备")
    if targets.wr_feasible_mask.device != outputs.wr_logits.device:
        raise ValueError("WR 标签和 logits 必须位于同一设备")
    if targets.wr_labeled_group_mask.device != outputs.wr_logits.device:
        raise ValueError("WR 分组标签掩码和 logits 必须位于同一设备")

    ww_selected = targets.ww_labeled_mask
    if ww_selected.any():
        selected_labels = targets.ww_labels[ww_selected]
        if torch.any((selected_labels != 0) & (selected_labels != 1)):
            raise ValueError("有标签的 WW target 必须为 0 或 1")
        ww_loss = F.binary_cross_entropy_with_logits(
            outputs.ww_logits[ww_selected],
            selected_labels.to(outputs.ww_logits.dtype),
            pos_weight=ww_pos_weight,
        )
        ww_examples = int(ww_selected.sum().item())
    else:
        ww_loss = outputs.ww_logits.sum() * 0.0
        ww_examples = 0

    group_count = targets.wr_labeled_group_mask.numel()
    if targets.wr_labeled_group_mask.dtype != torch.bool:
        raise ValueError("wr_labeled_group_mask 必须使用 torch.bool")
    if outputs.wr_group_indices.numel():
        if outputs.wr_group_indices.min().item() < 0 or outputs.wr_group_indices.max().item() >= group_count:
            raise IndexError("wr_group_indices 超出标签分组范围")
        all_denominator = _segment_logsumexp(
            outputs.wr_logits, outputs.wr_group_indices, group_count
        )
        # 未监督组不参与 loss，把它们的全部候选临时视为可行，避免空集合产生 NaN。
        safe_feasible = targets.wr_feasible_mask | ~targets.wr_labeled_group_mask[
            outputs.wr_group_indices
        ]
        feasible_scores = outputs.wr_logits.masked_fill(~safe_feasible, -torch.inf)
        feasible_numerator = _segment_logsumexp(
            feasible_scores, outputs.wr_group_indices, group_count
        )
        labeled = targets.wr_labeled_group_mask
        if labeled.any() and torch.any(~torch.isfinite(feasible_numerator[labeled])):
            raise ValueError("每个有标签的 WR 组必须至少包含一个可行候选")
        wr_loss = (
            (all_denominator[labeled] - feasible_numerator[labeled]).mean()
            if labeled.any()
            else outputs.wr_logits.sum() * 0.0
        )
        wr_groups = int(labeled.sum().item())
    else:
        if group_count != 0:
            raise ValueError("存在 WR 分组标签，但模型没有 WR 候选")
        wr_loss = outputs.wr_logits.sum() * 0.0
        wr_groups = 0

    return VeriStrongLoss(
        total=ww_loss + wr_weight * wr_loss,
        ww=ww_loss,
        wr=wr_loss,
        ww_examples=ww_examples,
        wr_groups=wr_groups,
    )


__all__ = [
    "VeriStrongLoss",
    "VeriStrongTargets",
    "build_veristrong_targets",
    "collate_veristrong_targets",
    "veristrong_decision_loss",
]
