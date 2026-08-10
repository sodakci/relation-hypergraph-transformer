"""将 VeriStrong 约束问题转换为四类节点的 PyTorch 异构图。"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Literal, Mapping

import torch
from torch import Tensor

from .graph import EdgePair
from .veristrong_data import VeriStrongProblem

VeriStrongNodeType = Literal["transaction", "key", "decision", "constraint"]

VERISTRONG_RELATION_ENDPOINTS: Final = MappingProxyType(
    {
        "SO": ("transaction", "transaction"),
        "SO_REV": ("transaction", "transaction"),
        "WR_FIXED": ("transaction", "transaction"),
        "WR_FIXED_REV": ("transaction", "transaction"),
        "WW_FIXED": ("transaction", "transaction"),
        "WW_FIXED_REV": ("transaction", "transaction"),
        "RW_FIXED": ("transaction", "transaction"),
        "RW_FIXED_REV": ("transaction", "transaction"),
        "READS": ("transaction", "key"),
        "READ_BY": ("key", "transaction"),
        "WRITES": ("transaction", "key"),
        "WRITTEN_BY": ("key", "transaction"),
        "WW_LEFT": ("transaction", "decision"),
        "WW_LEFT_REV": ("decision", "transaction"),
        "WW_RIGHT": ("transaction", "decision"),
        "WW_RIGHT_REV": ("decision", "transaction"),
        "WR_WRITER": ("transaction", "decision"),
        "WR_WRITER_REV": ("decision", "transaction"),
        "WR_READER": ("transaction", "decision"),
        "WR_READER_REV": ("decision", "transaction"),
        "DECISION_KEY": ("decision", "key"),
        "KEY_DECISION": ("key", "decision"),
        "MEMBER_OF": ("decision", "constraint"),
        "HAS_MEMBER": ("constraint", "decision"),
    }
)

VERISTRONG_RELATIONS: Final[tuple[str, ...]] = tuple(VERISTRONG_RELATION_ENDPOINTS)


@dataclass(frozen=True)
class VeriStrongNodeLayout:
    """Transaction、Key、Decision、Constraint 的全局拼接布局。"""

    transaction_count: int
    key_count: int
    decision_count: int
    constraint_count: int

    def count(self, node_type: VeriStrongNodeType) -> int:
        return {
            "transaction": self.transaction_count,
            "key": self.key_count,
            "decision": self.decision_count,
            "constraint": self.constraint_count,
        }[node_type]

    def offset(self, node_type: VeriStrongNodeType) -> int:
        return {
            "transaction": 0,
            "key": self.transaction_count,
            "decision": self.transaction_count + self.key_count,
            "constraint": self.transaction_count + self.key_count + self.decision_count,
        }[node_type]

    @property
    def total_nodes(self) -> int:
        return self.offset("constraint") + self.constraint_count


@dataclass(frozen=True)
class VeriStrongGraph:
    """供 VeriStrong 神经求解模型使用的完整张量输入。"""

    transaction_features: Tensor
    key_features: Tensor
    decision_features: Tensor
    constraint_features: Tensor
    relation_edges: Mapping[str, EdgePair]
    decision_left_transaction: Tensor
    decision_right_transaction: Tensor
    decision_constraint: Tensor
    decision_wr_group: Tensor
    ww_mask: Tensor
    wr_mask: Tensor
    transaction_ids: tuple[int, ...]
    key_ids: tuple[int, ...]
    ww_choice_ids: tuple[int, ...]
    wr_choice_ids: tuple[int, ...]
    # 批处理图的原始图数量；解码只能对 graph_count=1 的单图执行。
    graph_count: int = 1

    @property
    def layout(self) -> VeriStrongNodeLayout:
        return VeriStrongNodeLayout(
            self.transaction_features.size(0),
            self.key_features.size(0),
            self.decision_features.size(0),
            self.constraint_features.size(0),
        )

    @property
    def wr_group_count(self) -> int:
        return len(self.wr_choice_ids)

    def validate(self, relation_names: set[str] | None = None) -> None:
        features = {
            "transaction_features": (self.transaction_features, 16),
            "key_features": (self.key_features, 6),
            "decision_features": (self.decision_features, 16),
            "constraint_features": (self.constraint_features, 6),
        }
        device = self.transaction_features.device
        for name, (tensor, width) in features.items():
            if tensor.ndim != 2 or tensor.size(1) != width or not tensor.is_floating_point():
                raise ValueError(f"{name} 必须是形状 [N, {width}] 的浮点张量")
            if tensor.device != device:
                raise ValueError("所有节点特征必须位于同一设备")

        decision_count = self.layout.decision_count
        for name, tensor, dtype in (
            ("decision_left_transaction", self.decision_left_transaction, torch.long),
            ("decision_right_transaction", self.decision_right_transaction, torch.long),
            ("decision_constraint", self.decision_constraint, torch.long),
            ("decision_wr_group", self.decision_wr_group, torch.long),
            ("ww_mask", self.ww_mask, torch.bool),
            ("wr_mask", self.wr_mask, torch.bool),
        ):
            if tensor.ndim != 1 or tensor.numel() != decision_count or tensor.dtype != dtype:
                raise ValueError(f"{name} 的形状或 dtype 非法")
            if tensor.device != device:
                raise ValueError(f"{name} 与节点特征不在同一设备")
        if torch.any(self.ww_mask & self.wr_mask) or not torch.all(self.ww_mask | self.wr_mask):
            raise ValueError("每个 Decision 必须且只能属于 WW 或 WR")
        if decision_count:
            for tensor in (self.decision_left_transaction, self.decision_right_transaction):
                if tensor.min().item() < 0 or tensor.max().item() >= self.layout.transaction_count:
                    raise IndexError("Decision 包含非法 Transaction 局部编号")
            if self.decision_constraint.min().item() < 0 or self.decision_constraint.max().item() >= self.layout.constraint_count:
                raise IndexError("Decision 包含非法 Constraint 局部编号")
        if self.ww_mask.any() and not torch.all(self.decision_wr_group[self.ww_mask] == -1):
            raise ValueError("WW Decision 的 WR group 必须为 -1")
        if self.wr_mask.any():
            groups = self.decision_wr_group[self.wr_mask]
            if groups.min().item() < 0 or groups.max().item() >= self.wr_group_count:
                raise IndexError("WR Decision 包含非法分组编号")
            if set(groups.tolist()) != set(range(self.wr_group_count)):
                raise ValueError("每个 WR 分组必须至少包含一个候选 Decision")

        if len(self.transaction_ids) != self.layout.transaction_count:
            raise ValueError("transaction_ids 数量与 Transaction 节点数不一致")
        if len(self.key_ids) != self.layout.key_count:
            raise ValueError("key_ids 数量与 Key 节点数不一致")
        if len(self.ww_choice_ids) != int(self.ww_mask.sum().item()):
            raise ValueError("ww_choice_ids 数量与 WW Decision 数不一致")

        allowed = relation_names or set(VERISTRONG_RELATIONS)
        for relation, (source, target) in self.relation_edges.items():
            if relation not in allowed or relation not in VERISTRONG_RELATION_ENDPOINTS:
                raise KeyError(f"模型未配置关系 {relation!r}")
            if source.dtype != torch.long or target.dtype != torch.long:
                raise TypeError(f"{relation} 边索引必须使用 torch.long")
            if source.device != device or target.device != device:
                raise ValueError(f"{relation} 边索引与节点特征不在同一设备")
            if source.ndim != 1 or target.ndim != 1 or source.numel() != target.numel():
                raise ValueError(f"{relation} 的源、目标必须是等长一维张量")
            source_type, target_type = VERISTRONG_RELATION_ENDPOINTS[relation]
            for indices, node_type in ((source, source_type), (target, target_type)):
                lower = self.layout.offset(node_type)
                upper = lower + self.layout.count(node_type)
                if indices.numel() and (indices.min().item() < lower or indices.max().item() >= upper):
                    raise IndexError(f"{relation} 端点不属于要求的 {node_type} 节点区间")

    def to(self, device: torch.device | str) -> "VeriStrongGraph":
        """将图中的全部张量移动到目标设备。"""

        tensor_fields = {
            name: getattr(self, name).to(device)
            for name in (
                "transaction_features",
                "key_features",
                "decision_features",
                "constraint_features",
                "decision_left_transaction",
                "decision_right_transaction",
                "decision_constraint",
                "decision_wr_group",
                "ww_mask",
                "wr_mask",
            )
        }
        return replace(
            self,
            **tensor_fields,
            relation_edges={
                relation: (source.to(device), target.to(device))
                for relation, (source, target) in self.relation_edges.items()
            },
        )


def _reachability(nodes: tuple[int, ...], adjacency: Mapping[int, set[int]]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for start in nodes:
        reached = {start}
        stack = list(adjacency[start])
        while stack:
            node = stack.pop()
            if node in reached:
                continue
            reached.add(node)
            stack.extend(adjacency[node] - reached)
        result[start] = reached
    return result


def _strong_components(nodes: tuple[int, ...], adjacency: Mapping[int, set[int]]) -> dict[int, int]:
    """用 Kosaraju 算法计算可能边辅助图的 SCC 编号。"""

    visited: set[int] = set()
    order: list[int] = []
    for start in nodes:
        if start in visited:
            continue
        stack: list[tuple[int, bool]] = [(start, False)]
        while stack:
            node, exiting = stack.pop()
            if exiting:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            stack.extend((target, False) for target in adjacency[node] if target not in visited)

    reverse = {node: set() for node in nodes}
    for source, targets in adjacency.items():
        for target in targets:
            reverse[target].add(source)
    component: dict[int, int] = {}
    component_id = 0
    for start in reversed(order):
        if start in component:
            continue
        stack = [start]
        component[start] = component_id
        while stack:
            node = stack.pop()
            for target in reverse[node]:
                if target not in component:
                    component[target] = component_id
                    stack.append(target)
        component_id += 1
    return component


class _LocalEdges:
    def __init__(self) -> None:
        self.edges: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def add(self, relation: str, source: int, target: int) -> None:
        self.edges[relation].append((source, target))

    def tensors(self, layout: VeriStrongNodeLayout) -> dict[str, EdgePair]:
        result: dict[str, EdgePair] = {}
        for relation, pairs in self.edges.items():
            source_type, target_type = VERISTRONG_RELATION_ENDPOINTS[relation]
            source = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
            target = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
            result[relation] = (
                source + layout.offset(source_type),
                target + layout.offset(target_type),
            )
        return result


def build_veristrong_graph(problem: VeriStrongProblem) -> VeriStrongGraph:
    """从剪枝后的 VeriStrong 问题构造四类节点图和全部关系边。"""

    if not problem.pruning_consistent:
        raise ValueError("剪枝已证明该问题冲突，不能构造待求解图")

    transactions = tuple(sorted(problem.history.transactions, key=lambda tx: tx.transaction_id))
    transaction_ids = tuple(tx.transaction_id for tx in transactions)
    transaction_index = {transaction_id: index for index, transaction_id in enumerate(transaction_ids)}
    key_ids = tuple(problem.history.observed_keys)
    key_index = {key: index for index, key in enumerate(key_ids)}
    nodes = transaction_ids

    known_adjacency = problem.fixed_edges.adjacency(nodes)
    known_reachability = _reachability(nodes, known_adjacency)
    possible_adjacency = {node: set(targets) for node, targets in known_adjacency.items()}
    for choice in problem.remaining_ww_choices:
        possible_adjacency[choice.left_transaction].add(choice.right_transaction)
        possible_adjacency[choice.right_transaction].add(choice.left_transaction)
    for choice in problem.remaining_wr_choices:
        for writer in choice.candidate_writers:
            possible_adjacency[writer].add(choice.read_transaction)
    possible_component = _strong_components(nodes, possible_adjacency)

    typed_in = {kind: defaultdict(int) for kind in ("SO", "WR", "WW", "RW")}
    typed_out = {kind: defaultdict(int) for kind in ("SO", "WR", "WW", "RW")}
    total_degree = defaultdict(int)
    for kind, source, target, _ in problem.fixed_edges.typed_edges():
        typed_out[kind][source] += 1
        typed_in[kind][target] += 1
        total_degree[source] += 1
        total_degree[target] += 1

    session_by_id = {session.session_id: session for session in problem.history.sessions}
    transaction_by_id = problem.history.transaction_by_id
    transaction_features: list[list[float]] = []
    for transaction in transactions:
        session_length = len(session_by_id[transaction.session_id].transactions)
        denominator = max(1, session_length - 1)
        transaction_features.append(
            [
                math.log1p(sum(not event.is_write for event in transaction.events)),
                math.log1p(sum(event.is_write for event in transaction.events)),
                math.log1p(len(transaction.events)),
                math.log1p(len({event.key for event in transaction.events})),
                transaction.session_position / denominator,
                math.log1p(session_length),
                math.log1p(typed_in["SO"][transaction.transaction_id]),
                math.log1p(typed_out["SO"][transaction.transaction_id]),
                math.log1p(typed_in["WR"][transaction.transaction_id]),
                math.log1p(typed_out["WR"][transaction.transaction_id]),
                math.log1p(typed_in["WW"][transaction.transaction_id]),
                math.log1p(typed_out["WW"][transaction.transaction_id]),
                math.log1p(typed_in["RW"][transaction.transaction_id]),
                math.log1p(typed_out["RW"][transaction.transaction_id]),
                (sum(transaction.transaction_id in known_reachability[node] for node in nodes) - 1)
                / max(1, len(nodes) - 1),
                (len(known_reachability[transaction.transaction_id]) - 1) / max(1, len(nodes) - 1),
            ]
        )

    readers: dict[int, set[int]] = defaultdict(set)
    writers: dict[int, set[int]] = defaultdict(set)
    read_count = defaultdict(int)
    write_count = defaultdict(int)
    for transaction in transactions:
        if transaction.is_initial:
            continue
        for event in transaction.events:
            if event.is_write:
                writers[event.key].add(transaction.transaction_id)
                write_count[event.key] += 1
            else:
                readers[event.key].add(transaction.transaction_id)
                read_count[event.key] += 1
    key_features = [
        [
            math.log1p(len(readers[key])),
            math.log1p(len(writers[key])),
            math.log1p(read_count[key]),
            math.log1p(write_count[key]),
            math.log1p(read_count[key] + write_count[key]),
            math.log1p(len(readers[key] | writers[key])),
        ]
        for key in key_ids
    ]

    # 先统计每个事务参与多少个剩余候选，用作 Decision 的局部拥塞特征。
    occurrences = defaultdict(int)
    for choice in problem.remaining_ww_choices:
        occurrences[choice.left_transaction] += 1
        occurrences[choice.right_transaction] += 1
    for choice in problem.remaining_wr_choices:
        occurrences[choice.read_transaction] += len(choice.candidate_writers)
        for writer in choice.candidate_writers:
            occurrences[writer] += 1

    local_edges = _LocalEdges()
    relation_name = {"SO": "SO", "WR": "WR_FIXED", "WW": "WW_FIXED", "RW": "RW_FIXED"}
    for kind, source, target, _ in problem.fixed_edges.typed_edges():
        forward = relation_name[kind]
        source_local = transaction_index[source]
        target_local = transaction_index[target]
        local_edges.add(forward, source_local, target_local)
        local_edges.add(f"{forward}_REV", target_local, source_local)

    # Key 访问关系按事务去重；初始事务仍连接全部初始化 Key。
    for transaction in transactions:
        read_keys = {event.key for event in transaction.events if not event.is_write}
        write_keys = {event.key for event in transaction.events if event.is_write}
        tx_local = transaction_index[transaction.transaction_id]
        for key in sorted(read_keys):
            local_edges.add("READS", tx_local, key_index[key])
            local_edges.add("READ_BY", key_index[key], tx_local)
        for key in sorted(write_keys):
            local_edges.add("WRITES", tx_local, key_index[key])
            local_edges.add("WRITTEN_BY", key_index[key], tx_local)

    decision_features: list[list[float]] = []
    constraint_features: list[list[float]] = []
    decision_left: list[int] = []
    decision_right: list[int] = []
    decision_constraint: list[int] = []
    decision_wr_group: list[int] = []
    ww_mask: list[bool] = []
    ww_choice_ids: list[int] = []
    wr_choice_ids: list[int] = []

    def append_decision(
        *,
        is_ww: bool,
        left_id: int,
        right_id: int,
        keys: tuple[int, ...],
        group_size: int,
        value_writer_count: int,
        constraint_local: int,
        wr_group: int,
    ) -> int:
        decision_local = len(decision_features)
        left_tx = transaction_by_id[left_id]
        right_tx = transaction_by_id[right_id]
        left_position = left_tx.session_position / max(
            1, len(session_by_id[left_tx.session_id].transactions) - 1
        )
        right_position = right_tx.session_position / max(
            1, len(session_by_id[right_tx.session_id].transactions) - 1
        )
        decision_features.append(
            [
                float(is_ww),
                float(not is_ww),
                float(left_tx.session_id == right_tx.session_id),
                float(possible_component[left_id] == possible_component[right_id]),
                math.log1p(group_size),
                math.log1p(len(keys)),
                math.log1p(total_degree[left_id]),
                math.log1p(total_degree[right_id]),
                float(right_id in known_reachability[left_id]),
                float(left_id in known_reachability[right_id]),
                left_position - right_position,
                math.log1p(value_writer_count),
                float(any(key in problem.accesses[left_id].writes for key in keys)),
                float(any(key in problem.accesses[right_id].reads for key in keys)),
                math.log1p(occurrences[left_id] + occurrences[right_id]),
                float(left_tx.is_initial or right_tx.is_initial),
            ]
        )
        decision_left.append(transaction_index[left_id])
        decision_right.append(transaction_index[right_id])
        decision_constraint.append(constraint_local)
        decision_wr_group.append(wr_group)
        ww_mask.append(is_ww)
        for key in keys:
            local_edges.add("DECISION_KEY", decision_local, key_index[key])
            local_edges.add("KEY_DECISION", key_index[key], decision_local)
        local_edges.add("MEMBER_OF", decision_local, constraint_local)
        local_edges.add("HAS_MEMBER", constraint_local, decision_local)
        return decision_local

    for choice in problem.remaining_ww_choices:
        constraint_local = len(constraint_features)
        left_id, right_id = choice.left_transaction, choice.right_transaction
        invalid_count = int(left_id in known_reachability[right_id]) + int(
            right_id in known_reachability[left_id]
        )
        constraint_features.append(
            [1.0, 0.0, math.log1p(2), float(0 in (left_id, right_id)), invalid_count / 2.0, 0.0]
        )
        decision_local = append_decision(
            is_ww=True,
            left_id=left_id,
            right_id=right_id,
            keys=choice.keys,
            group_size=2,
            value_writer_count=0,
            constraint_local=constraint_local,
            wr_group=-1,
        )
        left_local, right_local = transaction_index[left_id], transaction_index[right_id]
        local_edges.add("WW_LEFT", left_local, decision_local)
        local_edges.add("WW_LEFT_REV", decision_local, left_local)
        local_edges.add("WW_RIGHT", right_local, decision_local)
        local_edges.add("WW_RIGHT_REV", decision_local, right_local)
        ww_choice_ids.append(choice.group_id)

    for wr_group, choice in enumerate(problem.remaining_wr_choices):
        constraint_local = len(constraint_features)
        invalid_count = sum(
            writer in known_reachability[choice.read_transaction]
            for writer in choice.candidate_writers
        )
        constraint_features.append(
            [
                0.0,
                1.0,
                math.log1p(len(choice.candidate_writers)),
                float(0 in choice.candidate_writers),
                invalid_count / max(1, len(choice.candidate_writers)),
                0.0,
            ]
        )
        wr_choice_ids.append(choice.group_id)
        for writer in choice.candidate_writers:
            decision_local = append_decision(
                is_ww=False,
                left_id=writer,
                right_id=choice.read_transaction,
                keys=(choice.key,),
                group_size=len(choice.candidate_writers),
                value_writer_count=len(choice.candidate_writers),
                constraint_local=constraint_local,
                wr_group=wr_group,
            )
            writer_local = transaction_index[writer]
            reader_local = transaction_index[choice.read_transaction]
            local_edges.add("WR_WRITER", writer_local, decision_local)
            local_edges.add("WR_WRITER_REV", decision_local, writer_local)
            local_edges.add("WR_READER", reader_local, decision_local)
            local_edges.add("WR_READER_REV", decision_local, reader_local)

    decision_count = len(decision_features)
    constraint_count = len(constraint_features)
    layout = VeriStrongNodeLayout(len(transactions), len(key_ids), decision_count, constraint_count)
    graph = VeriStrongGraph(
        transaction_features=torch.tensor(transaction_features, dtype=torch.float32),
        key_features=torch.tensor(key_features, dtype=torch.float32).reshape(len(key_ids), 6),
        decision_features=torch.tensor(decision_features, dtype=torch.float32).reshape(decision_count, 16),
        constraint_features=torch.tensor(constraint_features, dtype=torch.float32).reshape(constraint_count, 6),
        relation_edges=local_edges.tensors(layout),
        decision_left_transaction=torch.tensor(decision_left, dtype=torch.long),
        decision_right_transaction=torch.tensor(decision_right, dtype=torch.long),
        decision_constraint=torch.tensor(decision_constraint, dtype=torch.long),
        decision_wr_group=torch.tensor(decision_wr_group, dtype=torch.long),
        ww_mask=torch.tensor(ww_mask, dtype=torch.bool),
        wr_mask=torch.tensor([not value for value in ww_mask], dtype=torch.bool),
        transaction_ids=transaction_ids,
        key_ids=key_ids,
        ww_choice_ids=tuple(ww_choice_ids),
        wr_choice_ids=tuple(wr_choice_ids),
    )
    graph.validate()
    return graph


def collate_veristrong_graphs(graphs: list[VeriStrongGraph]) -> VeriStrongGraph:
    """按节点类型拼接多个图，同时正确更新全局边、事务和分组偏移。"""

    if not graphs:
        raise ValueError("至少需要一个图才能构造 batch")
    for graph in graphs:
        graph.validate()

    totals = VeriStrongNodeLayout(
        transaction_count=sum(graph.layout.transaction_count for graph in graphs),
        key_count=sum(graph.layout.key_count for graph in graphs),
        decision_count=sum(graph.layout.decision_count for graph in graphs),
        constraint_count=sum(graph.layout.constraint_count for graph in graphs),
    )
    type_offsets = {node_type: 0 for node_type in ("transaction", "key", "decision", "constraint")}
    relation_parts: dict[str, list[EdgePair]] = defaultdict(list)
    left_parts: list[Tensor] = []
    right_parts: list[Tensor] = []
    constraint_parts: list[Tensor] = []
    wr_group_parts: list[Tensor] = []
    wr_group_offset = 0

    for graph in graphs:
        layout = graph.layout
        for relation, (source, target) in graph.relation_edges.items():
            source_type, target_type = VERISTRONG_RELATION_ENDPOINTS[relation]
            source_local = source - layout.offset(source_type) + type_offsets[source_type]
            target_local = target - layout.offset(target_type) + type_offsets[target_type]
            relation_parts[relation].append(
                (
                    source_local + totals.offset(source_type),
                    target_local + totals.offset(target_type),
                )
            )
        left_parts.append(graph.decision_left_transaction + type_offsets["transaction"])
        right_parts.append(graph.decision_right_transaction + type_offsets["transaction"])
        constraint_parts.append(graph.decision_constraint + type_offsets["constraint"])
        groups = graph.decision_wr_group.clone()
        groups[graph.wr_mask] += wr_group_offset
        wr_group_parts.append(groups)
        wr_group_offset += graph.wr_group_count
        for node_type in type_offsets:
            type_offsets[node_type] += layout.count(node_type)

    relation_edges = {
        relation: (
            torch.cat([part[0] for part in parts]),
            torch.cat([part[1] for part in parts]),
        )
        for relation, parts in relation_parts.items()
    }
    batch = VeriStrongGraph(
        transaction_features=torch.cat([graph.transaction_features for graph in graphs]),
        key_features=torch.cat([graph.key_features for graph in graphs]),
        decision_features=torch.cat([graph.decision_features for graph in graphs]),
        constraint_features=torch.cat([graph.constraint_features for graph in graphs]),
        relation_edges=relation_edges,
        decision_left_transaction=torch.cat(left_parts),
        decision_right_transaction=torch.cat(right_parts),
        decision_constraint=torch.cat(constraint_parts),
        decision_wr_group=torch.cat(wr_group_parts),
        ww_mask=torch.cat([graph.ww_mask for graph in graphs]),
        wr_mask=torch.cat([graph.wr_mask for graph in graphs]),
        transaction_ids=tuple(value for graph in graphs for value in graph.transaction_ids),
        key_ids=tuple(value for graph in graphs for value in graph.key_ids),
        ww_choice_ids=tuple(value for graph in graphs for value in graph.ww_choice_ids),
        wr_choice_ids=tuple(value for graph in graphs for value in graph.wr_choice_ids),
        graph_count=sum(graph.graph_count for graph in graphs),
    )
    batch.validate()
    return batch


__all__ = [
    "VERISTRONG_RELATION_ENDPOINTS",
    "VERISTRONG_RELATIONS",
    "VeriStrongGraph",
    "VeriStrongNodeLayout",
    "build_veristrong_graph",
    "collate_veristrong_graphs",
]
