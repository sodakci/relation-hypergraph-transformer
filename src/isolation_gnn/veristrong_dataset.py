"""VeriStrong 历史文件发现和惰性 PyTorch Dataset。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from torch.utils.data import Dataset

from .veristrong_data import (
    VeriStrongProblem,
    construct_veristrong_problem,
    parse_dbcop_history,
)
from .veristrong_graph import VeriStrongGraph, build_veristrong_graph, collate_veristrong_graphs
from .veristrong_labels import VeriStrongLabels, load_veristrong_labels
from .veristrong_loss import VeriStrongTargets, collate_veristrong_targets


def discover_dbcop_histories(root: str | Path) -> tuple[Path, ...]:
    """递归发现并稳定排序所有 ``history.bincode`` 文件。"""

    root_path = Path(root)
    if root_path.is_file():
        return (root_path,)
    if not root_path.exists():
        raise FileNotFoundError(root_path)
    return tuple(sorted(root_path.rglob("history.bincode")))


@dataclass(frozen=True)
class VeriStrongSample:
    path: Path
    problem: VeriStrongProblem
    graph: VeriStrongGraph


@dataclass(frozen=True)
class LabeledVeriStrongSample:
    path: Path
    problem: VeriStrongProblem
    graph: VeriStrongGraph
    labels: VeriStrongLabels
    targets: VeriStrongTargets


class VeriStrongHistoryDataset(Dataset[VeriStrongSample]):
    """按需解析历史并构图；可选择在内存中缓存已访问样本。"""

    def __init__(
        self,
        paths: Iterable[str | Path],
        *,
        fast_prune: bool = True,
        cache: bool = False,
    ) -> None:
        self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("Dataset 至少需要一个历史文件")
        self.fast_prune = fast_prune
        self.cache = cache
        self._cache: dict[int, VeriStrongSample] = {}

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        fast_prune: bool = True,
        cache: bool = False,
    ) -> "VeriStrongHistoryDataset":
        return cls(discover_dbcop_histories(root), fast_prune=fast_prune, cache=cache)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> VeriStrongSample:
        if index < 0:
            index += len(self.paths)
        if index < 0 or index >= len(self.paths):
            raise IndexError(index)
        if index in self._cache:
            return self._cache[index]
        path = self.paths[index]
        history = parse_dbcop_history(path)
        problem = construct_veristrong_problem(history, fast_prune=self.fast_prune)
        if not problem.pruning_consistent:
            raise ValueError(f"历史在 fast pruning 阶段已经发生冲突：{path}")
        sample = VeriStrongSample(path=path, problem=problem, graph=build_veristrong_graph(problem))
        if self.cache:
            self._cache[index] = sample
        return sample


def collate_veristrong_samples(samples: list[VeriStrongSample]) -> VeriStrongGraph:
    """DataLoader 默认 collate_fn：训练只需要批处理后的张量图。"""

    return collate_veristrong_graphs([sample.graph for sample in samples])


class LabeledVeriStrongHistoryDataset(Dataset[LabeledVeriStrongSample]):
    """把历史与显式标签文件配对，适用于正式监督训练。"""

    def __init__(
        self,
        pairs: Iterable[tuple[str | Path, str | Path]],
        *,
        fast_prune: bool = True,
        cache: bool = False,
        require_veristrong_model_labels: bool = True,
    ) -> None:
        self.pairs = tuple((Path(history), Path(labels)) for history, labels in pairs)
        if not self.pairs:
            raise ValueError("Dataset 至少需要一对历史和标签文件")
        self.fast_prune = fast_prune
        self.cache = cache
        # 默认把求解器来源作为正式训练数据的硬约束。
        self.require_veristrong_model_labels = require_veristrong_model_labels
        self._cache: dict[int, LabeledVeriStrongSample] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> LabeledVeriStrongSample:
        if index < 0:
            index += len(self.pairs)
        if index < 0 or index >= len(self.pairs):
            raise IndexError(index)
        if index in self._cache:
            return self._cache[index]
        history_path, label_path = self.pairs[index]
        problem = construct_veristrong_problem(
            parse_dbcop_history(history_path), fast_prune=self.fast_prune
        )
        if not problem.pruning_consistent:
            raise ValueError(f"历史在 fast pruning 阶段已经发生冲突：{history_path}")
        graph = build_veristrong_graph(problem)
        labels = load_veristrong_labels(label_path, problem=problem)
        if self.require_veristrong_model_labels:
            labels.require_veristrong_model()
        sample = LabeledVeriStrongSample(
            path=history_path,
            problem=problem,
            graph=graph,
            labels=labels,
            targets=labels.to_targets(graph),
        )
        if self.cache:
            self._cache[index] = sample
        return sample


def collate_labeled_veristrong_samples(
    samples: list[LabeledVeriStrongSample],
) -> tuple[VeriStrongGraph, VeriStrongTargets]:
    """DataLoader collate_fn：同时拼接异构图和 WW/WR 标签。"""

    if not samples:
        raise ValueError("至少需要一个有标签样本")
    return (
        collate_veristrong_graphs([sample.graph for sample in samples]),
        collate_veristrong_targets([sample.targets for sample in samples]),
    )


__all__ = [
    "LabeledVeriStrongHistoryDataset",
    "LabeledVeriStrongSample",
    "VeriStrongHistoryDataset",
    "VeriStrongSample",
    "collate_labeled_veristrong_samples",
    "collate_veristrong_samples",
    "discover_dbcop_histories",
]
