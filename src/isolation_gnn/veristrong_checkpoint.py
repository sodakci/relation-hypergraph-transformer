"""VeriStrong 模型和训练集标准化统计的可移植 checkpoint。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .veristrong_model import VeriStrongDecisionNetwork
from .veristrong_normalize import VeriStrongFeatureNormalizer

CHECKPOINT_FORMAT = "isolation-gnn-veristrong-checkpoint-v1"


@dataclass(frozen=True)
class LoadedVeriStrongCheckpoint:
    model: VeriStrongDecisionNetwork
    normalizer: VeriStrongFeatureNormalizer | None
    epoch: int
    metadata: Mapping[str, Any]


def save_veristrong_checkpoint(
    path: str | Path,
    model: VeriStrongDecisionNetwork,
    *,
    normalizer: VeriStrongFeatureNormalizer | None = None,
    epoch: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """保存架构配置、权重和可选训练集标准化统计。"""

    if epoch < 0:
        raise ValueError("epoch 不能为负数")
    # metadata 限制为 JSON 基础类型，保证 ``weights_only=True`` 能安全恢复。
    safe_metadata = json.loads(json.dumps(dict(metadata or {}), ensure_ascii=False))
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_config": {
            "hidden_dim": model.hidden_dim,
            "processor_steps": model.processor_steps,
            "dropout": model.dropout,
            "relation_names": list(model.relation_names),
            "aggregation": model.aggregation,
        },
        "model_state": model.state_dict(),
        "normalizer_state": normalizer.state_dict() if normalizer is not None else None,
        "epoch": epoch,
        "metadata": safe_metadata,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def _normalizer_from_state(
    state: Mapping[str, torch.Tensor] | None,
) -> VeriStrongFeatureNormalizer | None:
    if state is None:
        return None
    names = (
        "transaction_mean",
        "transaction_std",
        "key_mean",
        "key_std",
        "decision_mean",
        "decision_std",
        "constraint_mean",
        "constraint_std",
    )
    missing = set(names) - set(state)
    if missing:
        raise ValueError(f"checkpoint 缺少标准化统计：{sorted(missing)}")
    normalizer = VeriStrongFeatureNormalizer(*(state[name] for name in names))
    normalizer.load_state_dict(dict(state), strict=True)
    return normalizer


def load_veristrong_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedVeriStrongCheckpoint:
    """安全加载仅含 Tensor 和基础类型的 checkpoint，并恢复可推理模型。"""

    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        found = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"不支持的 VeriStrong checkpoint 格式：{found!r}")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise TypeError("checkpoint 缺少 model_config")
    model = VeriStrongDecisionNetwork(
        hidden_dim=int(config["hidden_dim"]),
        processor_steps=int(config["processor_steps"]),
        dropout=float(config["dropout"]),
        relation_names=tuple(config["relation_names"]),
        aggregation=str(config.get("aggregation","mean_sum")),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    normalizer = _normalizer_from_state(payload.get("normalizer_state"))
    if normalizer is not None:
        normalizer.to(device)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata 必须是字典")
    return LoadedVeriStrongCheckpoint(
        model=model,
        normalizer=normalizer,
        epoch=int(payload.get("epoch", 0)),
        metadata=metadata,
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "LoadedVeriStrongCheckpoint",
    "load_veristrong_checkpoint",
    "save_veristrong_checkpoint",
]
