"""逐条运行 VeriStrong，并把 SAT model 直接采集为正式训练标签。"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from isolation_gnn import (
    ConstraintConstructionError,
    VeriStrongModelError,
    construct_veristrong_problem,
    discover_dbcop_histories,
    labels_from_veristrong_model,
    load_veristrong_labels,
    load_veristrong_solver_model,
    parse_dbcop_history,
    save_veristrong_labels,
)


def _run_veristrong(
    checker: Path,
    history: Path,
    model_output: Path,
    *,
    timeout_seconds: float | None,
) -> None:
    """运行官方 checker，并保留包含 node map、变量语义和最终 model 的完整输出。"""

    command = [
        str(checker),
        str(history),
        "--history-type",
        "dbcop",
        "--isolation-level",
        "ser",
        "--pruning",
        "fast",
        "--solver",
        "acyclic-minisat",
        "--log-level",
        "INFO",
    ]
    # VeriStrong 的 node map 和 MiniSat model 来自两个输出流；行缓冲可减少重排。
    stdbuf = shutil.which("stdbuf")
    if stdbuf is not None:
        command = [stdbuf, "-oL", "-eL", *command]

    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"VeriStrong checker 退出码为 {completed.returncode}，输出保留在 {model_output}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行 VeriStrong，为每个 SAT 历史直接采集 labels.json"
    )
    parser.add_argument("checker", type=Path, help="启用了 SAT model 导出的 VeriStrong checker")
    parser.add_argument("history_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--model-name", default="veristrong-model.log")
    parser.add_argument("--label-name", default="labels.json")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="按稳定排序从第几个历史开始，便于断点续跑（从 1 开始）",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="重新运行 checker，并替换同目录下已有的 model 和标签",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    checker = args.checker.resolve()
    if not checker.is_file():
        raise FileNotFoundError(checker)
    if not checker.stat().st_mode & 0o111:
        raise PermissionError(f"checker 不可执行：{checker}")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds 必须为正数")

    history_root = args.history_root.resolve()
    all_histories = list(discover_dbcop_histories(history_root))
    if args.start_index < 1 or args.start_index > len(all_histories) + 1:
        raise ValueError("start-index 超出历史数量范围")
    histories = all_histories[args.start_index - 1 :]
    if args.max_samples is not None:
        histories = histories[: args.max_samples]

    sat = unsat = invalid = failed = skipped = 0
    total_histories = len(all_histories)
    for index, history in enumerate(histories, start=args.start_index):
        relative_parent = (
            Path() if history_root.is_file() else history.parent.relative_to(history_root)
        )
        sample_output = args.output_root / relative_parent
        model_path = sample_output / args.model_name
        label_path = sample_output / args.label_name

        if label_path.exists() and not args.overwrite:
            try:
                existing_problem = construct_veristrong_problem(
                    parse_dbcop_history(history), fast_prune=True
                )
                existing = load_veristrong_labels(label_path, problem=existing_problem)
                existing.require_veristrong_model()
            except (OSError, ValueError) as exc:
                invalid += 1
                print(
                    f"[{index}/{total_histories}] NON_AUTHORITATIVE_LABEL {label_path}: {exc}; "
                    "请换输出目录或显式使用 --overwrite"
                )
                if args.fail_fast:
                    raise
                continue
            skipped += 1
            print(f"[{index}/{total_histories}] SKIP_VERIFIED_LABEL {label_path}")
            continue

        try:
            model = None
            if model_path.exists() and not args.overwrite:
                try:
                    # 支持断点续跑：完整 model 可直接复用，崩溃或中断日志则自动重跑。
                    model = load_veristrong_solver_model(model_path)
                except (OSError, VeriStrongModelError):
                    model = None
            if model is None:
                _run_veristrong(
                    checker,
                    history,
                    model_path,
                    timeout_seconds=args.timeout_seconds,
                )
                model = load_veristrong_solver_model(model_path)
            problem = construct_veristrong_problem(
                parse_dbcop_history(history), fast_prune=True
            )
            if not model.accepted:
                unsat += 1
                print(f"[{index}/{total_histories}] VERISTRONG_UNSAT {history}")
                continue
            labels = labels_from_veristrong_model(problem, model)
            save_veristrong_labels(labels, label_path)
        except subprocess.TimeoutExpired as exc:
            failed += 1
            print(f"[{index}/{total_histories}] TIMEOUT {history}: {exc}")
            if args.fail_fast:
                raise
            continue
        except (ConstraintConstructionError, VeriStrongModelError, OSError, RuntimeError, ValueError) as exc:
            failed += 1
            print(f"[{index}/{total_histories}] FAILED {history}: {exc}")
            if args.fail_fast:
                raise
            continue

        sat += 1
        print(f"[{index}/{total_histories}] VERISTRONG_MODEL_SAT -> {label_path}")

    print(
        f"完成：sat_labels={sat} unsat={unsat} invalid_existing={invalid} "
        f"failed={failed} skipped={skipped}"
    )


if __name__ == "__main__":
    main()
