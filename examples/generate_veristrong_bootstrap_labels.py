"""用 Python Beam Search 生成链路调试标签；这些标签不是正式训练真值。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from isolation_gnn import (
    ConstraintConstructionError,
    LABEL_PRODUCER_VALIDATED_WITNESS,
    build_veristrong_graph,
    construct_veristrong_problem,
    decode_with_beam,
    discover_dbcop_histories,
    labels_from_witness,
    load_veristrong_checkpoint,
    parse_dbcop_history,
    save_veristrong_labels,
    scores_from_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量生成非权威的 Python Beam 调试标签")
    parser.add_argument("history_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--max-expansions", type=int, default=100_000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.history_root.resolve()
    histories = list(discover_dbcop_histories(root))
    if args.max_samples is not None:
        histories = histories[: args.max_samples]
    loaded = (
        load_veristrong_checkpoint(args.checkpoint, device=args.device)
        if args.checkpoint
        else None
    )
    if loaded is not None:
        loaded.model.eval()

    certified = certified_unsat = unknown = skipped = 0
    for index, history_path in enumerate(histories, start=1):
        relative_parent = history_path.parent.relative_to(root)
        output = args.output_root / relative_parent / "labels.json"
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            problem = construct_veristrong_problem(
                parse_dbcop_history(history_path), fast_prune=True
            )
        except ConstraintConstructionError as exc:
            certified_unsat += 1
            print(f"[{index}/{len(histories)}] CERTIFIED_UNSAT {history_path}: {exc}")
            continue
        if not problem.pruning_consistent:
            certified_unsat += 1
            print(f"[{index}/{len(histories)}] CERTIFIED_UNSAT {history_path}")
            continue

        ww_scores: dict[int, float] = {}
        wr_scores: dict[int, dict[int, float]] = {}
        if loaded is not None:
            graph = build_veristrong_graph(problem).to(args.device)
            if loaded.normalizer is not None:
                graph = loaded.normalizer(graph)
            with torch.no_grad():
                outputs = loaded.model(graph)
            ww_scores, wr_scores = scores_from_outputs(graph, outputs)
        result = decode_with_beam(
            problem,
            ww_scores,
            wr_scores,
            beam_size=args.beam_size,
            max_expansions=args.max_expansions,
        )
        if result.validation is None or not result.validation.valid:
            unknown += 1
            print(f"[{index}/{len(histories)}] UNKNOWN {history_path}")
            continue
        labels = labels_from_witness(
            problem,
            result.ww_assignments,
            result.wr_assignments,
            source=str(history_path),
            producer=LABEL_PRODUCER_VALIDATED_WITNESS,
        )
        save_veristrong_labels(labels, output)
        certified += 1
        print(f"[{index}/{len(histories)}] DEBUG_SAT -> {output}")

    print(
        f"完成：debug_sat={certified} certified_unsat={certified_unsat} "
        f"unknown={unknown} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
