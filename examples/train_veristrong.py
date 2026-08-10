"""使用 history/label 配对数据训练 VeriStrong WW/WR 神经决策模型。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.monitor import Aggregation
from torch.utils.data import DataLoader

from isolation_gnn import (
    CachedLabeledVeriStrongHistoryDataset,
    LabeledVeriStrongHistoryDataset,
    VeriStrongDecisionNetwork,
    VeriStrongFeatureNormalizer,
    collate_labeled_veristrong_samples,
    discover_dbcop_histories,
    load_cached_veristrong_normalizer,
    normalizer_cache_path,
    save_cached_veristrong_normalizer,
    save_veristrong_checkpoint,
    veristrong_decision_loss,
)


def _resolve_training_device(requested: str) -> torch.device:
    """解析训练设备，并在 CUDA 不可用时立即给出明确错误。"""

    try:
        device = torch.device(requested)
    except RuntimeError as error:
        raise ValueError(f"无效的训练设备：{requested}") from error

    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(
            "已请求 CUDA 训练，但当前 PyTorch/系统无法使用 CUDA："
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}。"
            "请确认 NVIDIA GPU 已暴露给当前系统、驱动可用，并安装 CUDA 版 PyTorch。"
        )

    # 未写编号时固定使用第 0 张卡，便于日志和 checkpoint 复现实验环境。
    device_index = 0 if device.index is None else device.index
    if device_index >= torch.cuda.device_count():
        raise ValueError(
            f"请求的 GPU cuda:{device_index} 不存在，当前仅检测到 "
            f"{torch.cuda.device_count()} 张 GPU"
        )
    return torch.device("cuda", device_index)


def _discover_pairs(history_root: Path, label_root: Path, label_name: str) -> list[tuple[Path, Path]]:
    """按历史相对目录寻找标签，例如 ``<label-root>/<case>/labels.json``。"""

    histories = discover_dbcop_histories(history_root)
    pairs: list[tuple[Path, Path]] = []
    for history in histories:
        relative_parent = history.parent.relative_to(history_root)
        label = label_root / relative_parent / label_name
        if label.exists():
            pairs.append((history, label))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 VeriStrong 神经决策模型")
    parser.add_argument("history_root", type=Path)
    parser.add_argument("label_root", type=Path)
    parser.add_argument("--label-name", default="labels.json")
    parser.add_argument("--output", type=Path, default=Path("artifacts/veristrong.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="训练设备，默认使用第一张 NVIDIA GPU；如需 CPU，请显式传入 cpu",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--cache",
        action="store_true",
        help="把已加载的图保留在 CPU 内存中，供当前训练进程的后续 epoch 复用",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="跨训练进程复用的磁盘图和标准化统计缓存目录",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="忽略已有磁盘缓存并重新构建；必须同时指定 --cache-dir",
    )
    parser.add_argument(
        "--allow-non-veristrong-labels",
        action="store_true",
        help="仅用于链路调试；允许旧版或 Python witness 标签参与训练",
    )
    parser.add_argument(
        "--aggregation",
        choices=("mean_sum", "relation_gated"),
        default="mean_sum",
        help="不同 relation 消息之间的聚合方式",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs 和 batch-size 必须为正数")
    if args.rebuild_cache and args.cache_dir is None:
        raise ValueError("--rebuild-cache 必须与 --cache-dir 一起使用")
    device = _resolve_training_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"训练设备：{device}（{torch.cuda.get_device_name(device)}）")
    else:
        print(f"训练设备：{device}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    pairs = _discover_pairs(args.history_root.resolve(), args.label_root.resolve(), args.label_name)
    random.shuffle(pairs)
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise FileNotFoundError("没有找到 history.bincode 与标签 JSON 配对")
    print(f"训练样本数：{len(pairs)}")

    require_solver_labels = not args.allow_non_veristrong_labels
    cache_dir = args.cache_dir.resolve() if args.cache_dir is not None else None
    if cache_dir is not None:
        print("正在计算 history/labels 内容哈希并检查磁盘缓存……")
        dataset = CachedLabeledVeriStrongHistoryDataset(
            pairs,
            cache_dir,
            memory_cache=args.cache,
            require_veristrong_model_labels=require_solver_labels,
            rebuild=args.rebuild_cache,
        )
        print(
            f"磁盘图缓存：{dataset.existing_cache_count}/{len(dataset)} 条已存在；"
            f"dataset_key={dataset.dataset_key[:12]}"
        )
        normalizer = (
            None
            if args.rebuild_cache
            else load_cached_veristrong_normalizer(cache_dir, dataset.dataset_key)
        )
        if normalizer is not None:
            print(
                "标准化统计缓存命中："
                f"{normalizer_cache_path(cache_dir, dataset.dataset_key)}"
            )
    else:
        dataset = LabeledVeriStrongHistoryDataset(
            pairs,
            cache=args.cache,
            require_veristrong_model_labels=require_solver_labels,
        )
        normalizer = None

    if normalizer is None:
        print("开始构图并计算训练集标准化统计……")

        def training_graphs():
            for index in range(len(dataset)):
                graph = dataset[index].graph
                print(f"[预处理 {index + 1}/{len(dataset)}] {pairs[index][0]}")
                yield graph

        # 标准化统计流式累计，不会把全部节点特征额外拼成一个大张量。
        normalizer = VeriStrongFeatureNormalizer.fit_iterable(training_graphs())
        if cache_dir is not None:
            saved_normalizer = save_cached_veristrong_normalizer(
                cache_dir,
                dataset.dataset_key,
                normalizer,
            )
            print(
                f"磁盘缓存构建完成：命中={dataset.cache_hits}，"
                f"新建={dataset.cache_misses}；标准化统计={saved_normalizer}"
            )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_labeled_veristrong_samples,
    )
    model = VeriStrongDecisionNetwork(
        hidden_dim=args.hidden_dim,
        processor_steps=args.steps,
        dropout=args.dropout,
        aggregation=args.aggregation,
    ).to(device)
    normalizer.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        updates = 0
        for graph, targets in loader:
            # 图、标签和标准化器必须位于同一 GPU，避免运算时发生设备不一致。
            graph = normalizer(graph.to(device))
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = veristrong_decision_loss(model(graph), targets)
            if losses.ww_examples == 0 and losses.wr_groups == 0:
                continue
            losses.total.backward()
            optimizer.step()
            total_loss += losses.total.item()
            updates += 1
        if updates == 0:
            raise ValueError("所有标签均为空，无法训练")
        print(f"epoch={epoch:03d} loss={total_loss / updates:.6f} updates={updates}")

    save_veristrong_checkpoint(
        args.output,
        model,
        normalizer=normalizer,
        epoch=args.epochs,
        metadata={
            "history_root": str(args.history_root.resolve()),
            "label_root": str(args.label_root.resolve()),
            "sample_count": len(pairs),
            "seed": args.seed,
            "requires_veristrong_model_labels": require_solver_labels,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "dataset_cache_key": (
                dataset.dataset_key
                if isinstance(dataset, CachedLabeledVeriStrongHistoryDataset)
                else None
            ),
        },
    )
    print(f"checkpoint 已保存：{args.output}")


if __name__ == "__main__":
    main()
