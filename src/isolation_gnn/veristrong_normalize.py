"""只用训练集拟合的 VeriStrong 节点特征标准化器。"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from .veristrong_graph import VeriStrongGraph


class VeriStrongFeatureNormalizer(nn.Module):
    """标准化连续列，同时保持 one-hot、布尔列和方向标记不变。"""

    _CONTINUOUS_COLUMNS = {
        # 位置和可达率（4、14、15）保留原始比例；其余为 log1p 后的计数。
        "transaction": (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        "key": tuple(range(6)),
        # 方向位置差（10）保留在 [-1, 1]；类型、可达性和访问标记均不标准化。
        "decision": (4, 5, 6, 7, 11, 14),
        # 第 4 列是不可行比例，第 5 列预留为比例特征。
        "constraint": (2,),
    }

    def __init__(
        self,
        transaction_mean: Tensor,
        transaction_std: Tensor,
        key_mean: Tensor,
        key_std: Tensor,
        decision_mean: Tensor,
        decision_std: Tensor,
        constraint_mean: Tensor,
        constraint_std: Tensor,
    ) -> None:
        super().__init__()
        for name, tensor in (
            ("transaction_mean", transaction_mean),
            ("transaction_std", transaction_std),
            ("key_mean", key_mean),
            ("key_std", key_std),
            ("decision_mean", decision_mean),
            ("decision_std", decision_std),
            ("constraint_mean", constraint_mean),
            ("constraint_std", constraint_std),
        ):
            self.register_buffer(name, tensor.detach().clone().float())

    @staticmethod
    def _statistics(values: Tensor, columns: tuple[int, ...]) -> tuple[Tensor, Tensor]:
        width = values.size(1)
        mean = values.new_zeros(width)
        std = values.new_ones(width)
        if values.size(0) and columns:
            indices = torch.tensor(columns, dtype=torch.long, device=values.device)
            selected = values[:, indices]
            mean[indices] = selected.mean(dim=0)
            std[indices] = selected.std(dim=0, unbiased=False).clamp_min(1e-6)
        return mean, std

    @classmethod
    def fit(cls, graphs: Sequence[VeriStrongGraph]) -> "VeriStrongFeatureNormalizer":
        """只从传入图（应为训练集）计算均值和标准差。"""

        if not graphs:
            raise ValueError("至少需要一个训练图拟合标准化器")
        return cls.fit_iterable(graphs)

    @classmethod
    def fit_iterable(
        cls, graphs: Iterable[VeriStrongGraph]
    ) -> "VeriStrongFeatureNormalizer":
        """流式拟合统计量，避免把大量历史的全部节点特征同时放入内存。"""

        widths = {"transaction": 16, "key": 6, "decision": 16, "constraint": 6}
        sums = {name: torch.zeros(width, dtype=torch.float64) for name, width in widths.items()}
        square_sums = {
            name: torch.zeros(width, dtype=torch.float64) for name, width in widths.items()
        }
        counts = {name: 0 for name in widths}
        graph_count = 0
        for graph in graphs:
            graph.validate()
            graph_count += 1
            feature_groups = {
                "transaction": graph.transaction_features,
                "key": graph.key_features,
                "decision": graph.decision_features,
                "constraint": graph.constraint_features,
            }
            for name, values in feature_groups.items():
                values64 = values.detach().cpu().to(torch.float64)
                sums[name] += values64.sum(dim=0)
                square_sums[name] += values64.square().sum(dim=0)
                counts[name] += values64.size(0)
        if graph_count == 0:
            raise ValueError("至少需要一个训练图拟合标准化器")

        statistics: dict[str, tuple[Tensor, Tensor]] = {}
        for name, width in widths.items():
            mean = torch.zeros(width, dtype=torch.float32)
            std = torch.ones(width, dtype=torch.float32)
            if counts[name]:
                columns = torch.tensor(cls._CONTINUOUS_COLUMNS[name], dtype=torch.long)
                selected_mean = sums[name][columns] / counts[name]
                variance = square_sums[name][columns] / counts[name] - selected_mean.square()
                mean[columns] = selected_mean.float()
                std[columns] = variance.clamp_min(0).sqrt().clamp_min(1e-6).float()
            statistics[name] = mean, std
        return cls(
            statistics["transaction"][0],
            statistics["transaction"][1],
            statistics["key"][0],
            statistics["key"][1],
            statistics["decision"][0],
            statistics["decision"][1],
            statistics["constraint"][0],
            statistics["constraint"][1],
        )

    def transform(self, graph: VeriStrongGraph) -> VeriStrongGraph:
        """返回标准化后的新图，不修改原图。"""

        device = graph.transaction_features.device

        def normalize(values: Tensor, mean: Tensor, std: Tensor) -> Tensor:
            return (values - mean.to(device)) / std.to(device)

        result = replace(
            graph,
            transaction_features=normalize(
                graph.transaction_features, self.transaction_mean, self.transaction_std
            ),
            key_features=normalize(graph.key_features, self.key_mean, self.key_std),
            decision_features=normalize(
                graph.decision_features, self.decision_mean, self.decision_std
            ),
            constraint_features=normalize(
                graph.constraint_features, self.constraint_mean, self.constraint_std
            ),
        )
        result.validate()
        return result

    def forward(self, graph: VeriStrongGraph) -> VeriStrongGraph:
        return self.transform(graph)


__all__ = ["VeriStrongFeatureNormalizer"]
