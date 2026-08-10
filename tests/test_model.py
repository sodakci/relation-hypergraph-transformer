import torch

from isolation_gnn import (
    IsolationDecisionNetwork,
    RecurrentRelationalProcessor,
    decision_loss,
)
from isolation_gnn.synthetic import make_synthetic_graph


def test_aggregation_means_each_relation_before_summing() -> None:
    """验证聚合顺序是“关系内均值、关系间求和”。"""

    processor = RecurrentRelationalProcessor(["A", "B"], hidden_dim=1, dropout=0.0)
    with torch.no_grad():
        processor.relation_weights["A"].weight.fill_(1.0)
        processor.relation_weights["B"].weight.fill_(1.0)
    states = torch.tensor([[2.0], [4.0], [8.0]])
    edges = {
        "A": (torch.tensor([0, 1]), torch.tensor([2, 2])),
        "B": (torch.tensor([0]), torch.tensor([2])),
    }
    # A 关系贡献 mean(2, 4)=3，B 关系贡献 2，所以节点 2 最终收到 5。
    messages = processor.aggregate_messages(states, edges)
    assert torch.allclose(messages, torch.tensor([[0.0], [0.0], [5.0]]))


def test_forward_shapes_and_test_time_steps_override() -> None:
    """两个头的输出形状正确，且测试时可以覆盖 Processor 轮数。"""

    graph, _, _ = make_synthetic_graph(repeats=2)
    model = IsolationDecisionNetwork(hidden_dim=16, processor_steps=2, dropout=0.0)
    outputs = model(graph, processor_steps=3)
    assert outputs.ar_logits.shape == (4,)
    assert outputs.ww_logits.shape == (4,)
    assert outputs.ar_indices.tolist() == [0, 1, 4, 5]
    assert outputs.ww_indices.tolist() == [2, 3, 6, 7]


def test_masked_loss_ignores_unlabeled_decisions_and_backpropagates() -> None:
    """未标注 Decision 不参与 loss，其余样本仍能正常反向传播。"""

    graph, labels, labeled_mask = make_synthetic_graph(repeats=1)
    labeled_mask[3] = False
    model = IsolationDecisionNetwork(hidden_dim=8, processor_steps=1, dropout=0.0)
    losses = decision_loss(model(graph), labels, labeled_mask=labeled_mask, ww_weight=2.0)
    assert losses.ar_examples == 2
    assert losses.ww_examples == 1
    losses.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_empty_labeled_head_has_finite_zero_loss() -> None:
    """一个输出头没有任何标签时应返回有限的零损失。"""

    graph, labels, labeled_mask = make_synthetic_graph(repeats=1)
    labeled_mask[graph.ww_mask] = False
    model = IsolationDecisionNetwork(hidden_dim=8, processor_steps=0, dropout=0.0)
    losses = decision_loss(model(graph), labels, labeled_mask=labeled_mask)
    assert losses.ww_examples == 0
    assert losses.ww.item() == 0.0
    assert torch.isfinite(losses.total)
