"""在未见历史上评估 VeriStrong WW/WR 模型及端到端认证成功率。"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median

import torch

from isolation_gnn import (
    DecodeStatus,
    LabeledVeriStrongHistoryDataset,
    decode_with_beam,
    discover_dbcop_histories,
    load_veristrong_checkpoint,
    scores_from_outputs,
    validate_witness,
    veristrong_decision_loss,
)


def _discover_pairs(
    history_root: Path,
    label_root: Path,
    label_name: str,
) -> list[tuple[Path, Path]]:
    """按照历史相对目录寻找对应的权威标签。"""

    pairs: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    for history in discover_dbcop_histories(history_root):
        relative_parent = history.parent.relative_to(history_root)
        label = label_root / relative_parent / label_name
        if label.exists():
            pairs.append((history, label))
        else:
            missing.append(label)
    if missing:
        examples = "、".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"{len(missing)} 条历史缺少标签，例如：{examples}")
    return pairs


def _sync(device: torch.device) -> None:
    """CUDA 是异步执行的，计时边界处必须同步。"""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, object]:
    """从累计混淆矩阵计算 WW 二分类指标。"""

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "examples": tp + tn + fp + fn,
        "accuracy": _safe_ratio(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _timing_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"total_ms": 0.0, "mean_ms": 0.0, "median_ms": 0.0, "max_ms": 0.0}
    return {
        "total_ms": sum(values),
        "mean_ms": mean(values),
        "median_ms": median(values),
        "max_ms": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 VeriStrong 神经模型")
    parser.add_argument("history_root", type=Path)
    parser.add_argument("label_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--label-name", default="labels.json")
    parser.add_argument("--output", type=Path, default=Path("artifacts/veristrong-eval.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--max-expansions", type=int, default=100_000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-decode", action="store_true", help="只计算标签指标，不运行认证解码")
    args = parser.parse_args()

    if args.beam_size < 1 or args.max_expansions < 1:
        raise ValueError("beam-size 和 max-expansions 必须为正数")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已请求 CUDA 评估，但当前环境无法使用 CUDA")

    history_root = args.history_root.resolve()
    label_root = args.label_root.resolve()
    pairs = _discover_pairs(history_root, label_root, args.label_name)
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise FileNotFoundError("没有找到可评估的 history/labels 配对")

    loaded = load_veristrong_checkpoint(args.checkpoint, device=device)
    if loaded.normalizer is None:
        raise ValueError("checkpoint 没有训练集标准化统计，不能进行一致评估")
    loaded.model.eval()
    loaded.normalizer.eval()
    dataset = LabeledVeriStrongHistoryDataset(
        pairs,
        cache=False,
        require_veristrong_model_labels=True,
    )
    print(
        f"评估设备：{device}；checkpoint epoch={loaded.epoch}；"
        f"测试历史数：{len(dataset)}"
    )

    tp = tn = fp = fn = 0
    wr_correct = 0
    wr_groups = 0
    exact_histories = 0
    direct_valid_histories = 0
    losses: list[float] = []
    construct_times: list[float] = []
    prepare_times: list[float] = []
    model_times: list[float] = []
    decode_times: list[float] = []
    decode_statuses: Counter[str] = Counter()
    per_history: list[dict[str, object]] = []

    for index in range(len(dataset)):
        total_started = time.perf_counter()
        sample = dataset[index]
        constructed = time.perf_counter()
        sample.labels.require_veristrong_model()

        graph = sample.graph.to(device)
        targets = sample.targets.to(device)
        graph = loaded.normalizer(graph)
        _sync(device)
        prepared = time.perf_counter()
        model_started = prepared
        with torch.inference_mode():
            outputs = loaded.model(graph)
            losses_for_sample = veristrong_decision_loss(outputs, targets)
        _sync(device)
        modeled = time.perf_counter()
        ww_scores, wr_scores = scores_from_outputs(graph, outputs)

        sample_tp = sample_tn = sample_fp = sample_fn = 0
        for group_id, expected in sample.labels.ww_labels.items():
            predicted = ww_scores[group_id] >= 0.0
            if predicted and expected:
                sample_tp += 1
            elif not predicted and not expected:
                sample_tn += 1
            elif predicted:
                sample_fp += 1
            else:
                sample_fn += 1
        tp += sample_tp
        tn += sample_tn
        fp += sample_fp
        fn += sample_fn

        # 端到端直接 witness 必须覆盖全部 WR 组，即使以后评估部分标签也不能漏项。
        raw_wr_assignments = {
            group_id: max(candidate_scores, key=candidate_scores.__getitem__)
            for group_id, candidate_scores in wr_scores.items()
        }
        sample_wr_correct = 0
        sample_wr_groups = 0
        for group_id, feasible_writers in sample.labels.wr_feasible_writers.items():
            predicted_writer = raw_wr_assignments[group_id]
            sample_wr_correct += int(predicted_writer in feasible_writers)
            sample_wr_groups += 1
        wr_correct += sample_wr_correct
        wr_groups += sample_wr_groups

        raw_ww_assignments = {
            group_id: score >= 0.0 for group_id, score in ww_scores.items()
        }
        direct_validation = validate_witness(
            sample.problem,
            raw_ww_assignments,
            raw_wr_assignments,
        )
        direct_valid_histories += int(direct_validation.valid)
        exact_match = (
            sample_tp + sample_tn == len(sample.labels.ww_labels)
            and sample_wr_correct == sample_wr_groups
        )
        exact_histories += int(exact_match)

        decode_status: str | None = None
        expansions: int | None = None
        decode_started = time.perf_counter()
        if not args.skip_decode:
            result = decode_with_beam(
                sample.problem,
                ww_scores,
                wr_scores,
                beam_size=args.beam_size,
                max_expansions=args.max_expansions,
            )
            decode_status = result.status.value
            expansions = result.expansions
            decode_statuses[decode_status] += 1
        decoded = time.perf_counter()

        construct_ms = (constructed - total_started) * 1000
        prepare_ms = (prepared - constructed) * 1000
        model_ms = (modeled - model_started) * 1000
        decode_ms = (decoded - decode_started) * 1000
        construct_times.append(construct_ms)
        prepare_times.append(prepare_ms)
        model_times.append(model_ms)
        decode_times.append(decode_ms)
        losses.append(float(losses_for_sample.total.item()))
        relative_path = str(sample.path.relative_to(history_root))
        history_result = {
            "history": relative_path,
            "loss": losses[-1],
            "ww_examples": len(sample.labels.ww_labels),
            "ww_accuracy": _safe_ratio(
                sample_tp + sample_tn,
                sample_tp + sample_tn + sample_fp + sample_fn,
            ),
            "wr_groups": sample_wr_groups,
            "wr_top1_accuracy": _safe_ratio(sample_wr_correct, sample_wr_groups),
            "exact_label_match": exact_match,
            "direct_witness_valid": direct_validation.valid,
            "decode_status": decode_status,
            "decode_expansions": expansions,
            "graph": {
                "transactions": sample.graph.layout.transaction_count,
                "keys": sample.graph.layout.key_count,
                "decisions": sample.graph.layout.decision_count,
                "constraints": sample.graph.layout.constraint_count,
            },
            "timings_ms": {
                "construct": construct_ms,
                "prepare": prepare_ms,
                "model": model_ms,
                "decode": decode_ms,
            },
        }
        per_history.append(history_result)
        status_text = decode_status or "SKIPPED"
        print(
            f"[{index + 1}/{len(dataset)}] {status_text} "
            f"WW={history_result['ww_accuracy']!s} "
            f"WR={history_result['wr_top1_accuracy']!s} "
            f"model_ms={model_ms:.1f} {relative_path}"
        )

        # 大小差异显著的图连续评估时主动释放引用，降低 CUDA allocator 碎片风险。
        del graph, targets, outputs, losses_for_sample, sample

    summary = {
        "format": "isolation-gnn-veristrong-evaluation-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": loaded.epoch,
        "checkpoint_metadata": dict(loaded.metadata),
        "device": str(device),
        "sample_count": len(dataset),
        "mean_sample_loss": mean(losses),
        "ww": _binary_metrics(tp, tn, fp, fn),
        "wr": {
            "groups": wr_groups,
            "top1_correct": wr_correct,
            "top1_accuracy": _safe_ratio(wr_correct, wr_groups),
        },
        "exact_label_match": {
            "histories": exact_histories,
            "rate": _safe_ratio(exact_histories, len(dataset)),
        },
        "direct_witness": {
            "valid_histories": direct_valid_histories,
            "rate": _safe_ratio(direct_valid_histories, len(dataset)),
        },
        "beam_decode": {
            "skipped": args.skip_decode,
            "beam_size": args.beam_size,
            "max_expansions": args.max_expansions,
            "statuses": dict(sorted(decode_statuses.items())),
            "certified_sat_rate": (
                _safe_ratio(decode_statuses[DecodeStatus.CERTIFIED_SAT.value], len(dataset))
                if not args.skip_decode
                else None
            ),
        },
        "timings": {
            "construct": _timing_summary(construct_times),
            "prepare": _timing_summary(prepare_times),
            "model": _timing_summary(model_times),
            "decode": _timing_summary(decode_times),
        },
        "histories": per_history,
    }
    if not math.isfinite(summary["mean_sample_loss"]):
        raise ValueError("评估 loss 出现 NaN 或 Inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "histories"},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"完整评估结果已保存：{args.output}")


if __name__ == "__main__":
    main()
