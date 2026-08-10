"""可跨训练进程复用的 VeriStrong 图、标签张量和标准化统计缓存。"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .veristrong_data import construct_veristrong_problem, parse_dbcop_history
from .veristrong_graph import VeriStrongGraph, build_veristrong_graph
from .veristrong_labels import load_veristrong_labels
from .veristrong_loss import VeriStrongTargets
from .veristrong_normalize import VeriStrongFeatureNormalizer


# 构图特征或序列化字段发生语义变化时必须升级版本，使旧缓存自动失效。
SAMPLE_CACHE_FORMAT = "isolation-gnn-veristrong-tensor-cache-v1"
NORMALIZER_CACHE_FORMAT = "isolation-gnn-veristrong-normalizer-cache-v1"

_GRAPH_TENSOR_FIELDS = (
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
_NORMALIZER_STATE_FIELDS = (
    "transaction_mean",
    "transaction_std",
    "key_mean",
    "key_std",
    "decision_mean",
    "decision_std",
    "constraint_mean",
    "constraint_std",
)


def _update_digest_from_file(digest: object, marker: bytes, path: Path) -> None:
    """把文件类型、长度和内容加入哈希，避免不同字段拼接产生歧义。"""

    stat = path.stat()
    digest.update(marker)  # type: ignore[attr-defined]
    digest.update(stat.st_size.to_bytes(8, "little"))  # type: ignore[attr-defined]
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)  # type: ignore[attr-defined]


def veristrong_sample_cache_key(
    history_path: str | Path,
    label_path: str | Path,
    *,
    fast_prune: bool = True,
    require_veristrong_model_labels: bool = True,
) -> str:
    """根据输入文件内容和预处理选项生成稳定缓存键。"""

    history = Path(history_path)
    labels = Path(label_path)
    digest = hashlib.sha256()
    digest.update(SAMPLE_CACHE_FORMAT.encode("ascii"))
    digest.update(bytes((fast_prune, require_veristrong_model_labels)))
    _update_digest_from_file(digest, b"H", history)
    _update_digest_from_file(digest, b"L", labels)
    return digest.hexdigest()


def veristrong_dataset_cache_key(sample_keys: Iterable[str]) -> str:
    """用样本集合生成 normalizer 缓存键；样本迭代顺序不影响统计语义。"""

    keys = sorted(sample_keys)
    if not keys:
        raise ValueError("至少需要一个样本缓存键")
    digest = hashlib.sha256(NORMALIZER_CACHE_FORMAT.encode("ascii"))
    for key in keys:
        digest.update(b"\0")
        digest.update(key.encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CachedLabeledVeriStrongSample:
    """训练实际需要的最小缓存样本，不保存庞大的 Python 约束对象。"""

    path: Path
    graph: VeriStrongGraph
    targets: VeriStrongTargets


def _cpu_tensor(tensor: Tensor) -> Tensor:
    return tensor.detach().cpu().contiguous()


def _sample_payload(
    sample_key: str,
    graph: VeriStrongGraph,
    targets: VeriStrongTargets,
) -> dict[str, object]:
    return {
        "format": SAMPLE_CACHE_FORMAT,
        "sample_key": sample_key,
        "graph": {
            **{name: _cpu_tensor(getattr(graph, name)) for name in _GRAPH_TENSOR_FIELDS},
            "relation_edges": {
                relation: (_cpu_tensor(source), _cpu_tensor(target))
                for relation, (source, target) in graph.relation_edges.items()
            },
            "transaction_ids": list(graph.transaction_ids),
            "key_ids": list(graph.key_ids),
            "ww_choice_ids": list(graph.ww_choice_ids),
            "wr_choice_ids": list(graph.wr_choice_ids),
            "graph_count": graph.graph_count,
        },
        "targets": {
            "ww_labels": _cpu_tensor(targets.ww_labels),
            "ww_labeled_mask": _cpu_tensor(targets.ww_labeled_mask),
            "wr_feasible_mask": _cpu_tensor(targets.wr_feasible_mask),
            "wr_labeled_group_mask": _cpu_tensor(targets.wr_labeled_group_mask),
        },
    }


def _validate_targets(graph: VeriStrongGraph, targets: VeriStrongTargets) -> None:
    ww_count = int(graph.ww_mask.sum().item())
    wr_candidate_count = int(graph.wr_mask.sum().item())
    if targets.ww_labels.shape != (ww_count,):
        raise ValueError("缓存 WW 标签数量与图不一致")
    if targets.ww_labeled_mask.shape != (ww_count,):
        raise ValueError("缓存 WW 标签掩码数量与图不一致")
    if targets.wr_feasible_mask.shape != (wr_candidate_count,):
        raise ValueError("缓存 WR 候选标签数量与图不一致")
    if targets.wr_labeled_group_mask.shape != (graph.wr_group_count,):
        raise ValueError("缓存 WR 分组标签数量与图不一致")


def _sample_from_payload(
    path: Path,
    sample_key: str,
    payload: object,
) -> CachedLabeledVeriStrongSample:
    if not isinstance(payload, dict) or payload.get("format") != SAMPLE_CACHE_FORMAT:
        raise ValueError("缓存格式不受支持")
    if payload.get("sample_key") != sample_key:
        raise ValueError("缓存键与当前输入不一致")
    raw_graph = payload.get("graph")
    raw_targets = payload.get("targets")
    if not isinstance(raw_graph, dict) or not isinstance(raw_targets, dict):
        raise TypeError("缓存缺少 graph 或 targets")
    relation_edges = raw_graph.get("relation_edges")
    if not isinstance(relation_edges, dict):
        raise TypeError("缓存 relation_edges 必须是字典")

    graph = VeriStrongGraph(
        **{name: raw_graph[name] for name in _GRAPH_TENSOR_FIELDS},
        relation_edges={
            str(relation): (pair[0], pair[1])
            for relation, pair in relation_edges.items()
        },
        transaction_ids=tuple(int(value) for value in raw_graph["transaction_ids"]),
        key_ids=tuple(int(value) for value in raw_graph["key_ids"]),
        ww_choice_ids=tuple(int(value) for value in raw_graph["ww_choice_ids"]),
        wr_choice_ids=tuple(int(value) for value in raw_graph["wr_choice_ids"]),
        graph_count=int(raw_graph["graph_count"]),
    )
    targets = VeriStrongTargets(
        ww_labels=raw_targets["ww_labels"],
        ww_labeled_mask=raw_targets["ww_labeled_mask"],
        wr_feasible_mask=raw_targets["wr_feasible_mask"],
        wr_labeled_group_mask=raw_targets["wr_labeled_group_mask"],
    )
    graph.validate()
    _validate_targets(graph, targets)
    return CachedLabeledVeriStrongSample(path=path, graph=graph, targets=targets)


def _atomic_torch_save(payload: Mapping[str, object], destination: Path) -> None:
    """先写同目录临时文件再原子替换，避免中断留下半个缓存。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class CachedLabeledVeriStrongHistoryDataset(Dataset[CachedLabeledVeriStrongSample]):
    """优先读取逐样本磁盘缓存，未命中时严格解析历史和权威标签后写回。"""

    def __init__(
        self,
        pairs: Iterable[tuple[str | Path, str | Path]],
        cache_dir: str | Path,
        *,
        fast_prune: bool = True,
        memory_cache: bool = False,
        require_veristrong_model_labels: bool = True,
        rebuild: bool = False,
    ) -> None:
        self.pairs = tuple((Path(history), Path(labels)) for history, labels in pairs)
        if not self.pairs:
            raise ValueError("Dataset 至少需要一对历史和标签文件")
        self.cache_dir = Path(cache_dir)
        self.fast_prune = fast_prune
        self.memory_cache = memory_cache
        self.require_veristrong_model_labels = require_veristrong_model_labels
        self.rebuild = rebuild
        self.sample_keys = tuple(
            veristrong_sample_cache_key(
                history,
                labels,
                fast_prune=fast_prune,
                require_veristrong_model_labels=require_veristrong_model_labels,
            )
            for history, labels in self.pairs
        )
        self.dataset_key = veristrong_dataset_cache_key(self.sample_keys)
        self._memory_cache: dict[int, CachedLabeledVeriStrongSample] = {}
        self._rebuilt_indices: set[int] = set()
        self.cache_hits = 0
        self.cache_misses = 0

    def __len__(self) -> int:
        return len(self.pairs)

    def cache_path(self, index: int) -> Path:
        return self.cache_dir / "samples" / f"{self.sample_keys[index]}.pt"

    @property
    def existing_cache_count(self) -> int:
        return sum(self.cache_path(index).exists() for index in range(len(self)))

    def _build_sample(self, index: int) -> CachedLabeledVeriStrongSample:
        history_path, label_path = self.pairs[index]
        problem = construct_veristrong_problem(
            parse_dbcop_history(history_path),
            fast_prune=self.fast_prune,
        )
        if not problem.pruning_consistent:
            raise ValueError(f"历史在 fast pruning 阶段已经发生冲突：{history_path}")
        graph = build_veristrong_graph(problem)
        labels = load_veristrong_labels(label_path, problem=problem)
        if self.require_veristrong_model_labels:
            labels.require_veristrong_model()
        targets = labels.to_targets(graph)
        _atomic_torch_save(
            _sample_payload(self.sample_keys[index], graph, targets),
            self.cache_path(index),
        )
        return CachedLabeledVeriStrongSample(history_path, graph, targets)

    def __getitem__(self, index: int) -> CachedLabeledVeriStrongSample:
        if index < 0:
            index += len(self.pairs)
        if index < 0 or index >= len(self.pairs):
            raise IndexError(index)
        if index in self._memory_cache:
            return self._memory_cache[index]

        sample: CachedLabeledVeriStrongSample | None = None
        cache_path = self.cache_path(index)
        force_rebuild = self.rebuild and index not in self._rebuilt_indices
        if not force_rebuild and cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
                sample = _sample_from_payload(
                    self.pairs[index][0],
                    self.sample_keys[index],
                    payload,
                )
                self.cache_hits += 1
            except (
                AttributeError,
                EOFError,
                IndexError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                # 损坏或旧格式缓存按未命中处理，并由原子写覆盖。
                sample = None
        if sample is None:
            self.cache_misses += 1
            sample = self._build_sample(index)
            self._rebuilt_indices.add(index)
        if self.memory_cache:
            self._memory_cache[index] = sample
        return sample


def normalizer_cache_path(cache_dir: str | Path, dataset_key: str) -> Path:
    return Path(cache_dir) / "normalizers" / f"{dataset_key}.pt"


def load_cached_veristrong_normalizer(
    cache_dir: str | Path,
    dataset_key: str,
) -> VeriStrongFeatureNormalizer | None:
    """安全加载与完整训练样本集合绑定的标准化统计。"""

    path = normalizer_cache_path(cache_dir, dataset_key)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != NORMALIZER_CACHE_FORMAT:
            return None
        if payload.get("dataset_key") != dataset_key:
            return None
        state = payload.get("state")
        if not isinstance(state, dict) or set(_NORMALIZER_STATE_FIELDS) - set(state):
            return None
        normalizer = VeriStrongFeatureNormalizer(
            *(state[name] for name in _NORMALIZER_STATE_FIELDS)
        )
        normalizer.load_state_dict(state, strict=True)
        return normalizer
    except (
        AttributeError,
        EOFError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None


def save_cached_veristrong_normalizer(
    cache_dir: str | Path,
    dataset_key: str,
    normalizer: VeriStrongFeatureNormalizer,
) -> Path:
    """原子保存训练集标准化统计。"""

    path = normalizer_cache_path(cache_dir, dataset_key)
    state = {
        name: _cpu_tensor(tensor)
        for name, tensor in normalizer.state_dict().items()
    }
    _atomic_torch_save(
        {
            "format": NORMALIZER_CACHE_FORMAT,
            "dataset_key": dataset_key,
            "state": state,
        },
        path,
    )
    return path


__all__ = [
    "CachedLabeledVeriStrongHistoryDataset",
    "CachedLabeledVeriStrongSample",
    "NORMALIZER_CACHE_FORMAT",
    "SAMPLE_CACHE_FORMAT",
    "load_cached_veristrong_normalizer",
    "normalizer_cache_path",
    "save_cached_veristrong_normalizer",
    "veristrong_dataset_cache_key",
    "veristrong_sample_cache_key",
]
