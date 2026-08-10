"""项目命令行工具。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .veristrong_data import (
    ConstraintConstructionError,
    VeriStrongProblem,
    construct_veristrong_problem,
    export_problem_json,
    parse_dbcop_history,
    problem_summary,
)
from .veristrong_graph import build_veristrong_graph
from .veristrong_checkpoint import load_veristrong_checkpoint
from .veristrong_decode import DecodeResult, DecodeStatus, decode_with_beam, scores_from_outputs
from .veristrong_labels import (
    LABEL_PRODUCER_VALIDATED_WITNESS,
    labels_from_witness,
    problem_fingerprint,
    save_veristrong_labels,
)
from .veristrong_model import VeriStrongDecisionNetwork


def inspect_veristrong(argv: list[str] | None = None) -> None:
    """解析一个真实历史，输出剪枝、图张量和可选前向统计。"""

    parser = argparse.ArgumentParser(description="检查 VeriStrong DBCop 历史及神经图")
    parser.add_argument("history", help="history.bincode 路径")
    parser.add_argument("--no-fast-prune", action="store_true", help="仅固定 unit WR")
    parser.add_argument("--export-json", help="导出稳定调试 JSON")
    parser.add_argument("--forward", action="store_true", help="运行一次随机初始化模型前向")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args(argv)

    history = parse_dbcop_history(args.history)
    problem = construct_veristrong_problem(history, fast_prune=not args.no_fast_prune)
    summary = problem_summary(problem)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.export_json:
        export_problem_json(problem, args.export_json)
        print(f"已导出：{args.export_json}")
    if not problem.pruning_consistent:
        print("fast pruning 已发现冲突，不构造待求解图")
        return

    graph = build_veristrong_graph(problem)
    print(
        "graph:",
        f"transactions={graph.layout.transaction_count}",
        f"keys={graph.layout.key_count}",
        f"decisions={graph.layout.decision_count}",
        f"constraints={graph.layout.constraint_count}",
        f"relations={len(graph.relation_edges)}",
    )
    if args.forward:
        torch.manual_seed(7)
        model = VeriStrongDecisionNetwork(
            hidden_dim=args.hidden_dim,
            processor_steps=args.steps,
            dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            outputs = model(graph)
        print(f"outputs: ww={outputs.ww_logits.numel()} wr_candidates={outputs.wr_logits.numel()}")


def _prediction_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("history", help="history.bincode 路径")
    parser.add_argument("--checkpoint", help="训练后的 .pt checkpoint；省略时使用零分数基线")
    parser.add_argument("--device", default="cpu", help="例如 cpu 或 cuda")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--max-expansions", type=int, default=100_000)
    parser.add_argument(
        "--processor-steps",
        type=int,
        help="覆盖 checkpoint 中的消息传递轮数，便于规模外推理",
    )
    return parser


def _run_prediction(
    args: argparse.Namespace,
) -> tuple[VeriStrongProblem, DecodeResult, dict[str, object]]:
    """共享的解析、模型推理、约束解码和计时流程。"""

    started = time.perf_counter()
    history = parse_dbcop_history(args.history)
    problem = construct_veristrong_problem(history, fast_prune=True)
    constructed = time.perf_counter()
    if not problem.pruning_consistent:
        result = decode_with_beam(
            problem, {}, {}, beam_size=args.beam_size, max_expansions=args.max_expansions
        )
        return problem, result, {
            "construct_ms": (constructed - started) * 1000,
            "model_ms": 0.0,
            "decode_ms": 0.0,
            "checkpoint": args.checkpoint,
        }

    graph = build_veristrong_graph(problem)
    model_started = time.perf_counter()
    if args.checkpoint:
        loaded = load_veristrong_checkpoint(args.checkpoint, device=args.device)
        loaded.model.eval()
        graph_for_model = graph.to(args.device)
        if loaded.normalizer is not None:
            graph_for_model = loaded.normalizer(graph_for_model)
        with torch.no_grad():
            outputs = loaded.model(
                graph_for_model, processor_steps=args.processor_steps
            )
        ww_scores, wr_scores = scores_from_outputs(graph_for_model, outputs)
    else:
        # 零分数是可重复的无模型基线，也可用于验证完整数据链路。
        ww_scores, wr_scores = {}, {}
    modeled = time.perf_counter()
    result = decode_with_beam(
        problem,
        ww_scores,
        wr_scores,
        beam_size=args.beam_size,
        max_expansions=args.max_expansions,
    )
    finished = time.perf_counter()
    timings: dict[str, object] = {
        "construct_ms": (constructed - started) * 1000,
        "model_ms": (modeled - model_started) * 1000,
        "decode_ms": (finished - modeled) * 1000,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
    }
    return problem, result, timings


def _prediction_summary(
    problem: VeriStrongProblem,
    result: DecodeResult,
    timings: dict[str, object],
) -> dict[str, object]:
    return {
        "status": result.status.value,
        "certified": result.status != DecodeStatus.UNKNOWN,
        "ww_assignments": len(result.ww_assignments),
        "wr_assignments": len(result.wr_assignments),
        "expansions": result.expansions,
        "score": result.score,
        "problem_fingerprint": problem_fingerprint(problem),
        "problem": problem_summary(problem),
        "timings": timings,
    }


def predict_veristrong(argv: list[str] | None = None) -> None:
    """运行神经快速路径，并且只把通过独立校验的结果标为 CERTIFIED_SAT。"""

    parser = _prediction_parser("预测并认证一个 VeriStrong DBCop 历史")
    parser.add_argument("--witness-json", help="认证成功时导出完整 WW/WR assignment")
    args = parser.parse_args(argv)
    try:
        problem, result, timings = _run_prediction(args)
    except ConstraintConstructionError as exc:
        print(
            json.dumps(
                {
                    "status": DecodeStatus.CERTIFIED_UNSAT.value,
                    "certified": True,
                    "reason": str(exc),
                    "history": str(Path(args.history).resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    summary = _prediction_summary(problem, result, timings)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.witness_json and result.validation is not None and result.validation.valid:
        payload = {
            "format": "isolation-gnn-veristrong-witness-v1",
            "problem_fingerprint": problem_fingerprint(problem),
            "ww_assignments": {
                str(group): value for group, value in sorted(result.ww_assignments.items())
            },
            "wr_assignments": {
                str(group): writer for group, writer in sorted(result.wr_assignments.items())
            },
        }
        destination = Path(args.witness_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已导出认证 Witness：{destination}")


def label_veristrong(argv: list[str] | None = None) -> None:
    """从 Python Beam witness 生成仅供调试的非权威标签。"""

    parser = _prediction_parser("为一个 VeriStrong 历史生成认证单解标签")
    parser.add_argument("output", help="输出标签 JSON")
    args = parser.parse_args(argv)
    try:
        problem, result, timings = _run_prediction(args)
    except ConstraintConstructionError as exc:
        raise SystemExit(f"历史存在确定性约束冲突，不能生成 SAT 标签：{exc}") from exc
    if result.validation is None or not result.validation.valid:
        print(json.dumps(_prediction_summary(problem, result, timings), ensure_ascii=False, indent=2))
        raise SystemExit("未找到可认证 Witness，未生成标签；UNKNOWN 不能当作 UNSAT")
    labels = labels_from_witness(
        problem,
        result.ww_assignments,
        result.wr_assignments,
        source=str(Path(args.history).resolve()),
        producer=LABEL_PRODUCER_VALIDATED_WITNESS,
    )
    save_veristrong_labels(labels, args.output)
    print(f"已生成 Python witness 调试标签（正式训练默认拒绝）：{args.output}")


__all__ = ["inspect_veristrong", "label_veristrong", "predict_veristrong"]


if __name__ == "__main__":
    inspect_veristrong()
