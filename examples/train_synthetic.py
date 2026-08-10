"""过拟合一个小图，验证前向、反向传播和两个输出头。"""

from __future__ import annotations

import argparse

import torch

from isolation_gnn import IsolationDecisionNetwork, decision_loss
from isolation_gnn.synthetic import make_synthetic_graph


def main() -> None:
    """运行可配置的合成数据训练冒烟测试。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    # 固定随机种子，使参数初始化和训练结果可复现。
    torch.manual_seed(7)
    graph, labels, labeled_mask = make_synthetic_graph()
    model = IsolationDecisionNetwork(
        hidden_dim=args.hidden_dim,
        processor_steps=args.steps,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    initial_loss: float | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        # PyTorch 默认会累积梯度，每轮反向传播前必须清零。
        optimizer.zero_grad()
        losses = decision_loss(model(graph), labels, labeled_mask=labeled_mask)
        if initial_loss is None:
            initial_loss = losses.total.item()
        losses.total.backward()
        optimizer.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} total={losses.total.item():.4f} "
                f"ar={losses.ar.item():.4f} ww={losses.ww.item():.4f}"
            )

    # 评估阶段关闭 Dropout 和梯度记录，以确定性方式计算最终指标。
    model.eval()
    with torch.no_grad():
        outputs = model(graph)
        final_loss = decision_loss(outputs, labels, labeled_mask=labeled_mask).total.item()
        # BCE logit 的决策阈值为 0，等价于 sigmoid(logit) > 0.5。
        ar_accuracy = (
            (outputs.ar_logits > 0).float() == labels[outputs.ar_indices]
        ).float().mean().item()
        ww_accuracy = (
            (outputs.ww_logits > 0).float() == labels[outputs.ww_indices]
        ).float().mean().item()
    assert initial_loss is not None and final_loss < initial_loss
    print(
        f"initial={initial_loss:.4f} final={final_loss:.4f} "
        f"ar_accuracy={ar_accuracy:.3f} ww_accuracy={ww_accuracy:.3f}"
    )


if __name__ == "__main__":
    main()
