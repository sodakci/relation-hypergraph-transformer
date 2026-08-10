import pytest
import torch

from isolation_gnn import IsolationGraph, NodeLayout, globalize_relation_edges


def test_globalize_relation_edges_uses_type_offsets() -> None:
    """不同节点类型应使用各自的全局偏移量。"""

    layout = NodeLayout(transaction_count=3, key_count=2, decision_count=2)
    edges = globalize_relation_edges(
        {
            "WRITES": (torch.tensor([2]), torch.tensor([1])),
            "LEFT_TRANSACTION": (torch.tensor([1]), torch.tensor([0])),
        },
        layout,
    )
    assert edges["WRITES"][0].tolist() == [2]
    assert edges["WRITES"][1].tolist() == [4]
    assert edges["LEFT_TRANSACTION"][0].tolist() == [6]
    assert edges["LEFT_TRANSACTION"][1].tolist() == [0]


def test_graph_rejects_overlapping_decision_types() -> None:
    """同一个 Decision 不能同时交给 AR 和 WW 两个输出头。"""

    graph = IsolationGraph(
        transaction_features=torch.zeros(2, 16),
        key_features=torch.zeros(1, 6),
        decision_features=torch.zeros(1, 10),
        relation_edges={},
        decision_left_transaction=torch.tensor([0]),
        decision_right_transaction=torch.tensor([1]),
        ar_mask=torch.tensor([True]),
        ww_mask=torch.tensor([True]),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        graph.validate()


def test_graph_rejects_relation_endpoint_of_wrong_node_type() -> None:
    """即使全局编号未越界，关系端点的节点类型错误也必须被拒绝。"""

    graph = IsolationGraph(
        transaction_features=torch.zeros(2, 16),
        key_features=torch.zeros(1, 6),
        decision_features=torch.zeros(1, 10),
        # 全局编号 2 是 Key，但 LEFT_OF 的源节点必须是 Transaction。
        relation_edges={"LEFT_OF": (torch.tensor([2]), torch.tensor([3]))},
        decision_left_transaction=torch.tensor([0]),
        decision_right_transaction=torch.tensor([1]),
        ar_mask=torch.tensor([True]),
        ww_mask=torch.tensor([False]),
    )
    with pytest.raises(IndexError, match="source must reference transaction"):
        graph.validate()
