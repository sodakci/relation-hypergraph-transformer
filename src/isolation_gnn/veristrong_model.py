"""对应 VeriStrong WW/WR 实际求解变量的四节点类型网络。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import NodeEncoder, RecurrentRelationalProcessor
from .veristrong_graph import VERISTRONG_RELATIONS, VeriStrongGraph


class VeriStrongChoiceHead(nn.Module):
    """结合 Decision、左右事务和 Constraint 表示输出候选分数。"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 6, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        decision_state: Tensor,
        left_state: Tensor,
        right_state: Tensor,
        constraint_state: Tensor,
    ) -> Tensor:
        representation = torch.cat(
            (
                decision_state,
                left_state,
                right_state,
                left_state - right_state,
                left_state * right_state,
                constraint_state,
            ),
            dim=-1,
        )
        return self.network(representation).squeeze(-1)


@dataclass(frozen=True)
class VeriStrongOutputs:
    """WW 方向 logits 和 WR 候选分组 logits。"""

    ww_logits: Tensor
    wr_logits: Tensor
    ww_decision_indices: Tensor
    wr_decision_indices: Tensor
    wr_group_indices: Tensor
    wr_candidate_transaction_indices: Tensor


class VeriStrongDecisionNetwork(nn.Module):
    """使用共享循环 Processor 预测剪枝后剩余的 WW/WR 选择。"""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        processor_steps: int = 6,
        dropout: float = 0.1,
        relation_names: tuple[str, ...] | list[str] = VERISTRONG_RELATIONS,
        validate_inputs: bool = True,
        aggregation: str = "mean_sum",
    ) -> None:
        super().__init__()
        if processor_steps < 0:
            raise ValueError("processor_steps 不能为负数")
        self.hidden_dim = hidden_dim
        self.processor_steps = processor_steps
        self.dropout = dropout
        self.relation_names = tuple(relation_names)
        self.validate_inputs = validate_inputs
        self.aggregation= aggregation
        self.transaction_encoder = NodeEncoder(16, hidden_dim)
        self.key_encoder = NodeEncoder(6, hidden_dim)
        self.decision_encoder = NodeEncoder(16, hidden_dim)
        self.constraint_encoder = NodeEncoder(6, hidden_dim)
        self.processor = RecurrentRelationalProcessor(
            relation_names=self.relation_names,
            hidden_dim=hidden_dim,
            dropout=dropout,
            aggregation=aggregation,
        )
        # 两个头结构相同但参数独立，因为 WW 方向和 WR 来源选择语义不同。
        self.ww_head = VeriStrongChoiceHead(hidden_dim, dropout)
        self.wr_head = VeriStrongChoiceHead(hidden_dim, dropout)

    def forward(
        self,
        graph: VeriStrongGraph,
        *,
        processor_steps: int | None = None,
    ) -> VeriStrongOutputs:
        if self.validate_inputs:
            graph.validate(set(self.relation_names))
        steps = self.processor_steps if processor_steps is None else processor_steps
        if steps < 0:
            raise ValueError("processor_steps 不能为负数")

        transaction_state = self.transaction_encoder(graph.transaction_features)
        key_state = self.key_encoder(graph.key_features)
        decision_state = self.decision_encoder(graph.decision_features)
        constraint_state = self.constraint_encoder(graph.constraint_features)
        states = torch.cat(
            (transaction_state, key_state, decision_state, constraint_state), dim=0
        )
        for _ in range(steps):
            states = self.processor(states, graph.relation_edges)

        layout = graph.layout
        transaction_state = states[: layout.offset("key")]
        decision_state = states[
            layout.offset("decision") : layout.offset("constraint")
        ]
        constraint_state = states[layout.offset("constraint") :]
        left_state = transaction_state[graph.decision_left_transaction]
        right_state = transaction_state[graph.decision_right_transaction]
        group_state = constraint_state[graph.decision_constraint]
        ww_indices = torch.nonzero(graph.ww_mask, as_tuple=False).squeeze(-1)
        wr_indices = torch.nonzero(graph.wr_mask, as_tuple=False).squeeze(-1)

        return VeriStrongOutputs(
            ww_logits=self.ww_head(
                decision_state[ww_indices],
                left_state[ww_indices],
                right_state[ww_indices],
                group_state[ww_indices],
            ),
            wr_logits=self.wr_head(
                decision_state[wr_indices],
                left_state[wr_indices],
                right_state[wr_indices],
                group_state[wr_indices],
            ),
            ww_decision_indices=ww_indices,
            wr_decision_indices=wr_indices,
            wr_group_indices=graph.decision_wr_group[wr_indices],
            wr_candidate_transaction_indices=graph.decision_left_transaction[wr_indices],
        )


__all__ = [
    "VeriStrongChoiceHead",
    "VeriStrongDecisionNetwork",
    "VeriStrongOutputs",
]
