"""图输入校验，以及三类节点从局部编号到全局编号的转换。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch
from torch import Tensor

from .relations import NodeType, RELATION_ENDPOINTS

# 一条关系的边用两个等长一维张量表示：(源节点编号, 目标节点编号)。
EdgePair = tuple[Tensor, Tensor]


def _require_matrix(name: str, value: Tensor) -> None:
    """校验节点特征是否为二维浮点张量。"""

    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [nodes, features], got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")


def _require_vector(name: str, value: Tensor, length: int) -> None:
    """校验索引或掩码是否为指定长度的一维张量。"""

    if value.ndim != 1 or value.numel() != length:
        raise ValueError(f"{name} must have shape [{length}], got {tuple(value.shape)}")


@dataclass(frozen=True)
class NodeLayout:
    """记录拼接后的全局节点布局。

    全局节点顺序固定为 Transaction、Key、Decision。构图器和模型必须使用
    同一种顺序，否则边虽然可能没有越界，却会连接到错误类型的节点。
    """

    transaction_count: int
    key_count: int
    decision_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("transaction_count", self.transaction_count),
            ("key_count", self.key_count),
            ("decision_count", self.decision_count),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def key_offset(self) -> int:
        """Key 节点区间在总节点张量中的起始位置。"""

        return self.transaction_count

    @property
    def decision_offset(self) -> int:
        """Decision 节点区间在总节点张量中的起始位置。"""

        return self.transaction_count + self.key_count

    @property
    def total_nodes(self) -> int:
        """三类节点的总数。"""

        return self.decision_offset + self.decision_count

    def count(self, node_type: NodeType) -> int:
        return {
            "transaction": self.transaction_count,
            "key": self.key_count,
            "decision": self.decision_count,
        }[node_type]

    def offset(self, node_type: NodeType) -> int:
        return {
            "transaction": 0,
            "key": self.key_offset,
            "decision": self.decision_offset,
        }[node_type]

    def to_global(self, indices: Tensor, node_type: NodeType) -> Tensor:
        """将某一节点类型内部的局部编号转换为总节点张量的全局编号。"""

        if indices.dtype != torch.long:
            raise TypeError(f"{node_type} indices must use torch.long")
        if indices.ndim != 1:
            raise ValueError(f"{node_type} indices must be one-dimensional")
        count = self.count(node_type)
        if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= count):
            raise IndexError(f"{node_type} index is outside [0, {count})")
        return indices + self.offset(node_type)


def globalize_relation_edges(
    local_edges: Mapping[str, EdgePair],
    layout: NodeLayout,
) -> dict[str, EdgePair]:
    """按照关系两端的节点类型，将局部边编号转换为全局编号。"""

    result: dict[str, EdgePair] = {}
    for relation, endpoints in local_edges.items():
        if relation not in RELATION_ENDPOINTS:
            raise KeyError(f"unknown relation: {relation}")
        if len(endpoints) != 2:
            raise ValueError(f"{relation} must contain (source, target)")
        source, target = endpoints
        if source.device != target.device:
            raise ValueError(f"{relation} source and target must be on the same device")
        if source.ndim != 1 or target.ndim != 1 or source.numel() != target.numel():
            raise ValueError(f"{relation} source and target must be equal-length vectors")
        # 例如 WRITES 的源编号属于 Transaction，目标编号属于 Key；二者的
        # 偏移量不同，必须分别转换，不能统一加上同一个 offset。
        source_type, target_type = RELATION_ENDPOINTS[relation]
        result[relation] = (
            layout.to_global(source, source_type),
            layout.to_global(target, target_type),
        )
    return result


@dataclass(frozen=True)
class IsolationGraph:
    """一个异构图，或已经按节点类型拼接好的不连通图批次。

    relation_edges 使用全局节点编号；decision_left_transaction 和
    decision_right_transaction 则始终使用 Transaction 局部编号。
    """

    # 三类节点的特征形状分别为 [Nt, 16]、[Nk, 6]、[Nd, 10]（默认配置）。
    transaction_features: Tensor
    key_features: Tensor
    decision_features: Tensor
    # 关系边已经转换到拼接后的全局编号空间。
    relation_edges: Mapping[str, EdgePair]
    # 每个 Decision 对应的左右事务局部编号，形状均为 [Nd]。
    decision_left_transaction: Tensor
    decision_right_transaction: Tensor
    # 两个掩码必须互斥且完整覆盖全部 Decision。
    ar_mask: Tensor
    ww_mask: Tensor

    @property
    def layout(self) -> NodeLayout:
        """根据当前特征张量即时计算节点数量和偏移量。"""

        return NodeLayout(
            transaction_count=self.transaction_features.size(0),
            key_count=self.key_features.size(0),
            decision_count=self.decision_features.size(0),
        )

    def validate(self, relation_names: set[str] | None = None) -> None:
        """在模型计算前检查形状、设备、掩码和关系端点类型。"""

        # 所有节点特征必须是二维浮点张量，并位于同一个设备。
        _require_matrix("transaction_features", self.transaction_features)
        _require_matrix("key_features", self.key_features)
        _require_matrix("decision_features", self.decision_features)

        device = self.transaction_features.device
        feature_tensors = (self.key_features, self.decision_features)
        if any(tensor.device != device for tensor in feature_tensors):
            raise ValueError("all node features must be on the same device")

        # 分类头按 Decision 顺序读取左右事务，因此这两个索引必须与 Nd 等长。
        decision_count = self.layout.decision_count
        transaction_count = self.layout.transaction_count
        for name, value in (
            ("decision_left_transaction", self.decision_left_transaction),
            ("decision_right_transaction", self.decision_right_transaction),
        ):
            _require_vector(name, value, decision_count)
            if value.dtype != torch.long:
                raise TypeError(f"{name} must use torch.long")
            if value.device != device:
                raise ValueError(f"{name} must be on the node-feature device")
            if value.numel() and (value.min().item() < 0 or value.max().item() >= transaction_count):
                raise IndexError(f"{name} contains an invalid local transaction index")

        # 第一版仅支持 AR 和 WW；每个 Decision 必须且只能属于其中一种。
        for name, mask in (("ar_mask", self.ar_mask), ("ww_mask", self.ww_mask)):
            _require_vector(name, mask, decision_count)
            if mask.dtype != torch.bool:
                raise TypeError(f"{name} must use torch.bool")
            if mask.device != device:
                raise ValueError(f"{name} must be on the node-feature device")
        if torch.any(self.ar_mask & self.ww_mask):
            raise ValueError("AR and WW masks must not overlap")
        if not torch.all(self.ar_mask | self.ww_mask):
            raise ValueError("every decision must belong to exactly one of AR or WW")

        # 先检查边的一般合法性，再根据关系模式检查两端的具体节点类型。
        allowed = relation_names if relation_names is not None else set(RELATION_ENDPOINTS)
        for relation, (source, target) in self.relation_edges.items():
            if relation not in allowed:
                raise KeyError(f"relation {relation!r} is not configured in the model")
            if source.dtype != torch.long or target.dtype != torch.long:
                raise TypeError(f"{relation} edge indices must use torch.long")
            if source.device != device or target.device != device:
                raise ValueError(f"{relation} edge indices must be on the node-feature device")
            if source.ndim != 1 or target.ndim != 1 or source.numel() != target.numel():
                raise ValueError(f"{relation} source and target must be equal-length vectors")
            if source.numel() and (
                source.min().item() < 0
                or target.min().item() < 0
                or source.max().item() >= self.layout.total_nodes
                or target.max().item() >= self.layout.total_nodes
            ):
                raise IndexError(f"{relation} contains a global node index outside the graph")
            if relation in RELATION_ENDPOINTS:
                source_type, target_type = RELATION_ENDPOINTS[relation]
                for role, indices, node_type in (
                    ("source", source, source_type),
                    ("target", target, target_type),
                ):
                    lower = self.layout.offset(node_type)
                    upper = lower + self.layout.count(node_type)
                    if indices.numel() and (
                        indices.min().item() < lower or indices.max().item() >= upper
                    ):
                        raise IndexError(
                            f"{relation} {role} must reference {node_type} nodes "
                            f"in global range [{lower}, {upper})"
                        )

    def to(self, device: torch.device | str) -> "IsolationGraph":
        """将图内全部张量一起移动到 CPU、CUDA 等目标设备。"""

        return replace(
            self,
            transaction_features=self.transaction_features.to(device),
            key_features=self.key_features.to(device),
            decision_features=self.decision_features.to(device),
            relation_edges={
                relation: (source.to(device), target.to(device))
                for relation, (source, target) in self.relation_edges.items()
            },
            decision_left_transaction=self.decision_left_transaction.to(device),
            decision_right_transaction=self.decision_right_transaction.to(device),
            ar_mask=self.ar_mask.to(device),
            ww_mask=self.ww_mask.to(device),
        )
