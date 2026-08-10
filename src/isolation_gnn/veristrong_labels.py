"""VeriStrong 稳定标签格式、历史绑定校验和 Witness 标签转换。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .veristrong_data import VeriStrongProblem
from .veristrong_decode import validate_witness
from .veristrong_graph import VeriStrongGraph
from .veristrong_loss import VeriStrongTargets, build_veristrong_targets

LABEL_FORMAT = "isolation-gnn-veristrong-labels-v1"
# 标签来源是训练数据可信边界的一部分，不能只凭 witness 合法就冒充求解器真值。
LABEL_PRODUCER_VERISTRONG_MODEL = "veristrong_acyclic_minisat_model"
LABEL_PRODUCER_VALIDATED_WITNESS = "python_validated_witness"
LABEL_PRODUCER_LEGACY = "legacy_unspecified"


def problem_fingerprint(problem: VeriStrongProblem) -> str:
    """对约束语义计算稳定 SHA-256，防止标签误配到另一个历史。"""

    payload = {
        "transactions": list(problem.transaction_ids),
        "fixed_edges": [
            [kind, source, target, sorted(keys)]
            for kind, source, target, keys in problem.fixed_edges.typed_edges()
        ],
        "remaining_ww": [
            [
                choice.group_id,
                choice.left_transaction,
                choice.right_transaction,
                list(choice.keys),
            ]
            for choice in problem.remaining_ww_choices
        ],
        "remaining_wr": [
            [
                choice.group_id,
                choice.key,
                choice.value,
                choice.read_transaction,
                list(choice.candidate_writers),
            ]
            for choice in problem.remaining_wr_choices
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VeriStrongLabels:
    """以原始 group/transaction ID 表示的部分或完整监督标签。"""

    ww_labels: Mapping[int, bool]
    wr_feasible_writers: Mapping[int, frozenset[int]]
    label_kind: str = "mixed"
    fingerprint: str | None = None
    source: str | None = None
    producer: str = LABEL_PRODUCER_LEGACY

    def __post_init__(self) -> None:
        object.__setattr__(self, "ww_labels", {int(group): bool(value) for group, value in self.ww_labels.items()})
        object.__setattr__(
            self,
            "wr_feasible_writers",
            {
                int(group): frozenset(int(writer) for writer in writers)
                for group, writers in self.wr_feasible_writers.items()
            },
        )

    def validate(self, problem: VeriStrongProblem) -> None:
        """检查 ID、WR 候选集合和可选问题指纹。"""

        if self.fingerprint is not None and self.fingerprint != problem_fingerprint(problem):
            raise ValueError("标签指纹与当前剪枝后 VeriStrong 问题不一致")
        ww_by_id = {choice.group_id: choice for choice in problem.remaining_ww_choices}
        wr_by_id = {choice.group_id: choice for choice in problem.remaining_wr_choices}
        unknown_ww = set(self.ww_labels) - set(ww_by_id)
        unknown_wr = set(self.wr_feasible_writers) - set(wr_by_id)
        if unknown_ww:
            raise ValueError(f"标签包含未知 WW group：{sorted(unknown_ww)}")
        if unknown_wr:
            raise ValueError(f"标签包含未知 WR group：{sorted(unknown_wr)}")
        for group_id, writers in self.wr_feasible_writers.items():
            if not writers:
                raise ValueError(f"WR group {group_id} 的可行 writer 集合不能为空")
            candidates = set(wr_by_id[group_id].candidate_writers)
            if not writers <= candidates:
                raise ValueError(
                    f"WR group {group_id} 含非候选 writer：{sorted(writers - candidates)}"
                )

    @property
    def is_veristrong_model_label(self) -> bool:
        """标签是否直接来自 VeriStrong Acyclic-MiniSat 的 SAT model。"""

        return self.producer == LABEL_PRODUCER_VERISTRONG_MODEL

    def require_veristrong_model(self) -> None:
        """正式训练前检查权威来源，防止旧 Beam 标签静默混入。"""

        if not self.is_veristrong_model_label:
            raise ValueError(
                "正式训练只接受 VeriStrong SAT model 标签；"
                f"当前 producer={self.producer!r}，source={self.source!r}"
            )

    def to_targets(self, graph: VeriStrongGraph) -> VeriStrongTargets:
        return build_veristrong_targets(
            graph,
            ww_labels=dict(self.ww_labels),
            wr_feasible_writers={
                group: set(writers) for group, writers in self.wr_feasible_writers.items()
            },
        )


def labels_from_witness(
    problem: VeriStrongProblem,
    ww_assignments: Mapping[int, bool],
    wr_assignments: Mapping[int, int],
    *,
    source: str | None = None,
    producer: str = LABEL_PRODUCER_VALIDATED_WITNESS,
) -> VeriStrongLabels:
    """仅在完整 assignment 通过独立校验后生成单解标签。"""

    validation = validate_witness(problem, ww_assignments, wr_assignments)
    if not validation.valid:
        details = "; ".join(validation.errors[:5])
        raise ValueError(f"不能从无效 Witness 生成标签：{details}")
    labels = VeriStrongLabels(
        ww_labels={
            choice.group_id: bool(ww_assignments[choice.group_id])
            for choice in problem.remaining_ww_choices
        },
        wr_feasible_writers={
            choice.group_id: frozenset((int(wr_assignments[choice.group_id]),))
            for choice in problem.remaining_wr_choices
        },
        label_kind="single_solution",
        fingerprint=problem_fingerprint(problem),
        source=source,
        producer=producer,
    )
    labels.validate(problem)
    return labels


def save_veristrong_labels(labels: VeriStrongLabels, path: str | Path) -> None:
    """按稳定排序写出适合审阅和版本控制的 JSON。"""

    payload = {
        "format": LABEL_FORMAT,
        "label_kind": labels.label_kind,
        "problem_fingerprint": labels.fingerprint,
        "source": labels.source,
        "producer": labels.producer,
        "ww_labels": {
            str(group): value for group, value in sorted(labels.ww_labels.items())
        },
        "wr_feasible_writers": {
            str(group): sorted(writers)
            for group, writers in sorted(labels.wr_feasible_writers.items())
        },
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_veristrong_labels(
    path: str | Path,
    *,
    problem: VeriStrongProblem | None = None,
) -> VeriStrongLabels:
    """读取标签 JSON；提供 ``problem`` 时同时执行严格绑定校验。"""

    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != LABEL_FORMAT:
        found = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"不支持的 VeriStrong 标签格式：{found!r}")
    raw_ww = payload.get("ww_labels", {})
    raw_wr = payload.get("wr_feasible_writers", {})
    if not isinstance(raw_ww, dict) or not isinstance(raw_wr, dict):
        raise TypeError("ww_labels 和 wr_feasible_writers 必须是 JSON object")
    if any(type(value) is not bool for value in raw_ww.values()):
        raise TypeError("每个 WW 标签必须是 JSON boolean")
    if any(
        not isinstance(writers, list)
        or any(type(writer) is not int for writer in writers)
        for writers in raw_wr.values()
    ):
        raise TypeError("每个 WR 可行集合必须是整数数组")
    try:
        labels = VeriStrongLabels(
            ww_labels={int(group): value for group, value in raw_ww.items()},
            wr_feasible_writers={
                int(group): frozenset(writers) for group, writers in raw_wr.items()
            },
            label_kind=str(payload.get("label_kind", "mixed")),
            fingerprint=payload.get("problem_fingerprint"),
            source=payload.get("source") or str(source_path.resolve()),
            producer=str(payload.get("producer", LABEL_PRODUCER_LEGACY)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"标签包含非法 group ID：{source_path}") from exc
    if problem is not None:
        labels.validate(problem)
    return labels


__all__ = [
    "LABEL_FORMAT",
    "LABEL_PRODUCER_LEGACY",
    "LABEL_PRODUCER_VALIDATED_WITNESS",
    "LABEL_PRODUCER_VERISTRONG_MODEL",
    "VeriStrongLabels",
    "labels_from_witness",
    "load_veristrong_labels",
    "problem_fingerprint",
    "save_veristrong_labels",
]
