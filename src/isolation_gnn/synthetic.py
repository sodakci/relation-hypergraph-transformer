"""供训练示例和冒烟测试使用的小型确定性图。"""

from __future__ import annotations

import torch
from torch import Tensor

from .graph import IsolationGraph, NodeLayout, globalize_relation_edges


def _long(values: list[int]) -> Tensor:
    """把 Python 整数列表转换为图索引要求的 torch.long 张量。"""

    return torch.tensor(values, dtype=torch.long)


def make_synthetic_graph(repeats: int = 4) -> tuple[IsolationGraph, Tensor, Tensor]:
    """构造一个把方向标签编码在路径特征中的玩具图。

    该数据仅用于验证前向、反向和双头 loss 能否正常工作，不能用它衡量
    模型的真实泛化性能，也不能据此比较 GNN 与 MLP。
    """

    if repeats < 1:
        raise ValueError("repeats must be positive")
    transaction_count = repeats * 4
    decision_count = repeats * 4
    key_count = repeats * 2

    # 每四个 Transaction 组成一个互不连通的小组。
    transaction_features = torch.zeros(transaction_count, 16)
    transaction_features[:, 2] = torch.arange(transaction_count).remainder(4) + 1
    transaction_features[:, 4] = torch.arange(transaction_count).remainder(4)
    transaction_features[:, 15] = torch.arange(transaction_count).remainder(2).float()

    key_features = torch.zeros(key_count, 6)
    key_features[:, 0:2] = 2
    key_features[:, 4] = 4
    key_features[:, 5] = 2

    # 每组前两个 Decision 属于 AR，后两个属于 WW。
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0] * repeats)
    decision_features = torch.zeros(decision_count, 10)
    ar_mask = torch.tensor([True, True, False, False] * repeats)
    ww_mask = ~ar_mask
    decision_features[:, 0] = ar_mask.float()
    decision_features[:, 1] = ww_mask.float()
    # 特征 8/9 模拟“左到右有路径/右到左有路径”，故意让小样本容易过拟合。
    decision_features[:, 8] = labels
    decision_features[:, 9] = 1.0 - labels

    # left/right 保存 Transaction 局部编号；Decision 分类头会直接使用它们取状态。
    left = _long(
        [base + offset for base in range(0, transaction_count, 4) for offset in (0, 1, 0, 2)]
    )
    right = _long(
        [base + offset for base in range(0, transaction_count, 4) for offset in (1, 2, 2, 3)]
    )
    decisions = torch.arange(decision_count, dtype=torch.long)
    keys = _long([base + offset for base in range(0, key_count, 2) for offset in (0, 0, 1, 1)])
    tx_for_access = _long(
        [base + offset for base in range(0, transaction_count, 4) for offset in (0, 1, 2, 3)]
    )

    so_source = _long([base + offset for base in range(0, transaction_count, 4) for offset in (0, 1, 2)])
    so_target = so_source + 1
    # 此处所有边仍使用各自节点类型内部的局部编号。
    local_edges = {
        "SO": (so_source, so_target),
        "SO_REV": (so_target, so_source),
        "READS": (tx_for_access, keys),
        "READ_BY": (keys, tx_for_access),
        "WRITES": (tx_for_access, keys),
        "WRITTEN_BY": (keys, tx_for_access),
        "LEFT_OF": (left, decisions),
        "RIGHT_OF": (right, decisions),
        "LEFT_TRANSACTION": (decisions, left),
        "RIGHT_TRANSACTION": (decisions, right),
    }
    layout = NodeLayout(transaction_count, key_count, decision_count)
    graph = IsolationGraph(
        transaction_features=transaction_features,
        key_features=key_features,
        decision_features=decision_features,
        # 只有 relation_edges 需要转换成三类节点拼接后的全局编号。
        relation_edges=globalize_relation_edges(local_edges, layout),
        decision_left_transaction=left,
        decision_right_transaction=right,
        ar_mask=ar_mask,
        ww_mask=ww_mask,
    )
    labeled_mask = torch.ones(decision_count, dtype=torch.bool)
    return graph, labels, labeled_mask
