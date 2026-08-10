"""用于预测 AR/WW 方向的关系感知循环图神经网络。"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Mapping

import torch
from torch import AggregationType, Tensor, nn

from .graph import EdgePair, IsolationGraph
from .relations import DEFAULT_RELATIONS


class NodeEncoder(nn.Module):
    """将某一节点类型的原始特征映射到统一隐藏维度。"""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.encoder(features)


class DecisionHead(nn.Module):
    """读取 Decision 及其左右事务表示，输出一个未归一化 logit。"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 5, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        decision_state: Tensor,
        left_transaction_state: Tensor,
        right_transaction_state: Tensor,
    ) -> Tensor:
        # 五部分分别表达决策上下文、左右事务、方向差异和逐维交互。
        # 默认 hidden_dim=128，因此拼接后的维度为 5 * 128 = 640。
        pair_state = torch.cat(
            (
                decision_state,
                left_transaction_state,
                right_transaction_state,
                left_transaction_state - right_transaction_state,
                left_transaction_state * right_transaction_state,
            ),
            dim=-1,
        )
        return self.network(pair_state).squeeze(-1)


class RecurrentRelationalProcessor(nn.Module):
    """单轮关系消息处理器；同一组参数会被循环复用多轮。"""

    def __init__(
        self,
        relation_names: tuple[str, ...] | list[str],
        hidden_dim: int,
        dropout: float = 0.1,
        aggregation: str ="mean_sum",
    ) -> None:
        super().__init__()
        if len(set(relation_names)) != len(relation_names):
            raise ValueError("relation_names must be unique")
        self.hidden_dim = hidden_dim
        self.relation_names = tuple(relation_names)
        self.aggregation = aggregation
        if aggregation not in ["mean_sum","relation_gated"]:
            raise ValueError(f"unsupported aggregation: {aggregation}")
        self.relation_to_index = {
            relation: index
            for index,relation in enumerate(self.relation_names)
        }
        self.num_relations = len(self.relation_names)
        # 不同关系语义不同，因此每种关系使用独立的线性变换。
        self.relation_weights = nn.ModuleDict(
            {
                relation: nn.Linear(hidden_dim, hidden_dim, bias=False)
                for relation in self.relation_names
            }
        )
        if self.aggregation == "relation_gated":
            self.relation_embeddings = nn.Embedding(
                self.num_relations,
                hidden_dim,
            )
            #用均值为 0、标准差为 0.02 的正态分布（高斯分布）随机数，原地（in-place）覆盖掉关系嵌入层原有的权重参数
            nn.init.normal_(
                self.relation_embeddings.weight,
                mean = 0.0,
                std =  0.02,
            )
            self.relation_gate = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Tanh(), 
                nn.Linear(hidden_dim,1),
            )
        # GRU 保留旧状态，避免多轮传播时直接覆盖节点已有信息。
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.message_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def aggregate_messages(
        self,
        node_states: Tensor,
        relation_edges: Mapping[str, EdgePair],
    ) -> Tensor:
        """每种关系内部先求均值，再按照配置执行关系间聚合。"""

        #messages = torch.zeros_like(node_states)
        num_nodes = node_states.size(0)
        relation_messages = node_states.new_zeros(
            num_nodes,
            self.num_relations,
            self.hidden_dim,
        )

        relation_mask = torch.zeros(
            num_nodes,
            self.num_relations,
            dtype=torch.bool,
            device=node_states.device,
        )
        for relation, (source, target) in relation_edges.items():
            if relation not in self.relation_weights:
                raise KeyError(f"relation {relation!r} is not configured")
            if source.numel() == 0:
                continue

            # 对当前关系的源节点状态做关系专属变换，再按目标节点累加。
            relation_sum = torch.zeros_like(node_states)
            relation_sum.index_add_(
                0,
                target,
                self.relation_weights[relation](node_states[source]),
            )
            # 单独统计当前关系中每个目标节点的入边数，用于关系内 mean。
            relation_count = torch.zeros(
                node_states.size(0),
                1,
                device=node_states.device,
                dtype=node_states.dtype,
            )
            relation_count.index_add_(
                0,
                target,
                torch.ones(
                    target.numel(),
                    1,
                    device=node_states.device,
                    dtype=node_states.dtype,
                ),
            )
            # clamp_min 避免无入边节点除以零；这些节点在本关系中的消息保持零。
            #messages = messages + relation_sum / relation_count.clamp_min(1.0)
            relation_mean=(
                relation_sum / relation_count.clamp_min(1.0)
            )
            relation_index = self.relation_to_index[relation]
            relation_messages[:,relation_index,:] = relation_mean
            relation_mask[target,relation_index] = True
        if self.aggregation == "mean_sum":
            return relation_messages.sum(dim=1)   
        if self.aggregation == "relation_gated":
            # 逐关系计算 gate 分数，避免展开 [N, R, D] 和 [N, R, 3D] 的大张量。
            # 每个节点只对实际有入边的关系计算 gate，其余保持 -inf。
            relation_scores = torch.full(
                (num_nodes, self.num_relations),
                torch.finfo(node_states.dtype).min,
                device=node_states.device,
                dtype=node_states.dtype,
            )

            relation_ids = torch.arange(
                self.num_relations,
                device=node_states.device,
            )

            for r in range(self.num_relations):
                has_msg = relation_mask[:, r]
                if not has_msg.any():
                    continue

                n_msg = has_msg.sum().item()

                gate_input = torch.cat(
                    [
                        node_states[has_msg],
                        relation_messages[has_msg, r],
                        self.relation_embeddings(relation_ids[r])
                        .unsqueeze(0)
                        .expand(n_msg, -1),
                    ],
                    dim=-1,
                )

                relation_scores[has_msg, r] = (
                    self.relation_gate(gate_input).squeeze(-1)
                )

            has_message = relation_mask.any(dim=1)

            relation_weights = torch.zeros_like(relation_scores)

            relation_weights[has_message] = torch.softmax(
                relation_scores[has_message],
                dim=1,
            )

            messages = (
                relation_weights.unsqueeze(-1)
                * relation_messages
            ).sum(dim=1)

            return messages  

        raise RuntimeError(
            f"unexpected aggregation: {self.aggregation}"
        )

    def forward(
        self,
        node_states: Tensor,
        relation_edges: Mapping[str, EdgePair],
    ) -> Tensor:
        # 一轮更新顺序：消息归一化 -> GRU 更新 -> FFN 残差 -> 输出归一化。
        messages = self.message_norm(self.aggregate_messages(node_states, relation_edges))
        updated = self.gru(messages, node_states)
        updated = updated + self.ffn(updated)
        return self.output_norm(updated)


@dataclass(frozen=True)
class DecisionOutputs:
    """双头输出及其在原始 Decision 数组中的局部编号。"""

    ar_logits: Tensor
    ww_logits: Tensor
    ar_indices: Tensor
    ww_indices: Tensor


class IsolationDecisionNetwork(nn.Module):
    """Transaction + Key + Decision 三类节点的第一版预测网络。"""

    def __init__(
        self,
        transaction_dim: int = 16,
        key_dim: int = 6,
        decision_dim: int = 10,
        relation_names: tuple[str, ...] | list[str] = DEFAULT_RELATIONS,
        hidden_dim: int = 128,
        processor_steps: int = 6,
        dropout: float = 0.1,
        validate_inputs: bool = True,
    ) -> None:
        super().__init__()
        if processor_steps < 0:
            raise ValueError("processor_steps cannot be negative")
        self.transaction_dim = transaction_dim
        self.key_dim = key_dim
        self.decision_dim = decision_dim
        self.processor_steps = processor_steps
        self.validate_inputs = validate_inputs
        self.relation_names = tuple(relation_names)

        # 三类原始特征维数不同，分别编码后才能拼到同一个节点状态张量。
        self.transaction_encoder = NodeEncoder(transaction_dim, hidden_dim)
        self.key_encoder = NodeEncoder(key_dim, hidden_dim)
        self.decision_encoder = NodeEncoder(decision_dim, hidden_dim)
        self.processor = RecurrentRelationalProcessor(
            relation_names=self.relation_names,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        # AR 与 WW 语义不同，使用两个独立分类头而不是共享最后几层。
        self.ar_head = DecisionHead(hidden_dim, dropout)
        self.ww_head = DecisionHead(hidden_dim, dropout)

    def _validate_feature_dimensions(self, graph: IsolationGraph) -> None:
        """确保输入特征宽度与构造模型时声明的维度一致。"""

        expected = {
            "transaction_features": self.transaction_dim,
            "key_features": self.key_dim,
            "decision_features": self.decision_dim,
        }
        for name, width in expected.items():
            actual = getattr(graph, name).size(1)
            if actual != width:
                raise ValueError(f"{name} must have width {width}, got {actual}")

    def forward(
        self,
        graph: IsolationGraph,
        *,
        processor_steps: int | None = None,
    ) -> DecisionOutputs:
        """编码图并返回 AR/WW logits。

        processor_steps 可在测试时覆盖训练轮数，用于评估 6/8/10/12 轮
        消息传递的规模外泛化能力。
        """

        if self.validate_inputs:
            graph.validate(set(self.relation_names))
            self._validate_feature_dimensions(graph)

        steps = self.processor_steps if processor_steps is None else processor_steps
        if steps < 0:
            raise ValueError("processor_steps cannot be negative")

        # 按 Transaction -> Key -> Decision 的顺序形成全局节点状态张量。
        transaction_state = self.transaction_encoder(graph.transaction_features)
        key_state = self.key_encoder(graph.key_features)
        decision_state = self.decision_encoder(graph.decision_features)
        all_states = torch.cat((transaction_state, key_state, decision_state), dim=0)

        # 每轮调用的是同一个 processor，因此所有轮共享关系权重、GRU 和 FFN。
        for _ in range(steps):
            all_states = self.processor(all_states, graph.relation_edges)

        # 消息传播结束后，按全局布局切回分类头需要的节点类型。
        layout = graph.layout
        transaction_state = all_states[: layout.key_offset]
        decision_state = all_states[layout.decision_offset :]
        left_state = transaction_state[graph.decision_left_transaction]
        right_state = transaction_state[graph.decision_right_transaction]
        # 两个头只处理各自类型的 Decision，indices 用来与原标签重新对齐。
        ar_indices = torch.nonzero(graph.ar_mask, as_tuple=False).squeeze(-1)
        ww_indices = torch.nonzero(graph.ww_mask, as_tuple=False).squeeze(-1)

        return DecisionOutputs(
            ar_logits=self.ar_head(
                decision_state[ar_indices], left_state[ar_indices], right_state[ar_indices]
            ),
            ww_logits=self.ww_head(
                decision_state[ww_indices], left_state[ww_indices], right_state[ww_indices]
            ),
            ar_indices=ar_indices,
            ww_indices=ww_indices,
        )
