"""从 VeriStrong 导出的 Acyclic-MiniSat model 批量生成正式训练标签。"""

from __future__ import annotations

import argparse
from pathlib import Path

from isolation_gnn import (
    ConstraintConstructionError,
    VeriStrongModelError,
    construct_veristrong_problem,
    discover_dbcop_histories,
    labels_from_veristrong_model,
    load_veristrong_solver_model,
    parse_dbcop_history,
    save_veristrong_labels,
)


def _model_path(history: Path, history_root: Path, model_root: Path, name: str) -> Path:
    """让 model 目录与 history 目录使用同一套相对层级。"""

    if model_root.is_file():
        return model_root
    relative_parent = (
        Path() if history_root.is_file() else history.parent.relative_to(history_root)
    )
    return model_root / relative_parent / name


def main() -> None:
    parser = argparse.ArgumentParser(description="从 VeriStrong SAT model 生成权威单解标签")
    parser.add_argument("history_root", type=Path)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--model-name",
        default="model.log",
        help="每个历史相对目录中的 VeriStrong model 文件名",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="遇到缺失、格式错误或模型不一致时立即终止",
    )
    args = parser.parse_args()

    history_root = args.history_root.resolve()
    model_root = args.model_root.resolve()
    histories = list(discover_dbcop_histories(history_root))
    if args.max_samples is not None:
        histories = histories[: args.max_samples]
    if model_root.is_file() and len(histories) != 1:
        raise ValueError("model_root 是单个文件时，history_root 也必须只包含一个历史")

    imported = unsat = missing = invalid = skipped = 0
    for index, history_path in enumerate(histories, start=1):
        relative_parent = (
            Path()
            if history_root.is_file()
            else history_path.parent.relative_to(history_root)
        )
        model_path = _model_path(history_path, history_root, model_root, args.model_name)
        output_path = args.output_root / relative_parent / "labels.json"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if not model_path.exists():
            missing += 1
            print(f"[{index}/{len(histories)}] MISSING_MODEL {model_path}")
            if args.fail_fast:
                raise FileNotFoundError(model_path)
            continue

        try:
            problem = construct_veristrong_problem(
                parse_dbcop_history(history_path), fast_prune=True
            )
            model = load_veristrong_solver_model(model_path)
            if not model.accepted:
                unsat += 1
                print(f"[{index}/{len(histories)}] VERISTRONG_UNSAT {history_path}")
                continue
            labels = labels_from_veristrong_model(problem, model)
            save_veristrong_labels(labels, output_path)
        except (ConstraintConstructionError, VeriStrongModelError, ValueError) as exc:
            invalid += 1
            print(f"[{index}/{len(histories)}] INVALID_MODEL {history_path}: {exc}")
            if args.fail_fast:
                raise
            continue

        imported += 1
        print(f"[{index}/{len(histories)}] VERISTRONG_MODEL_SAT -> {output_path}")

    print(
        f"完成：imported={imported} veristrong_unsat={unsat} missing_model={missing} "
        f"invalid_model={invalid} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
