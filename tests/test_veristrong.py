from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch

from isolation_gnn import (
    CachedLabeledVeriStrongHistoryDataset,
    DbcopHeader,
    DbcopHistory,
    DecodeStatus,
    Event,
    LABEL_PRODUCER_VERISTRONG_MODEL,
    Session,
    Transaction,
    VeriStrongDecisionNetwork,
    VeriStrongFeatureNormalizer,
    VeriStrongModelError,
    VeriStrongTargets,
    build_veristrong_graph,
    build_veristrong_targets,
    collate_veristrong_targets,
    collate_veristrong_graphs,
    construct_veristrong_problem,
    decode_with_beam,
    labels_from_veristrong_model,
    parse_dbcop_history,
    parse_veristrong_model_text,
    problem_summary,
    labels_from_witness,
    load_cached_veristrong_normalizer,
    load_veristrong_checkpoint,
    load_veristrong_labels,
    save_veristrong_checkpoint,
    save_veristrong_labels,
    save_cached_veristrong_normalizer,
    validate_witness,
    veristrong_decision_loss,
)


def _i64(value: int) -> bytes:
    return struct.pack("<q", value)


def _string(value: str) -> bytes:
    encoded = value.encode()
    return _i64(len(encoded)) + encoded


def _event(is_write: bool, key: int, value: int, success: bool = True) -> bytes:
    return bytes((is_write,)) + _i64(key) + _i64(value) + bytes((success,))


def _transaction(events: list[bytes], committed: bool = True) -> bytes:
    return _i64(len(events)) + b"".join(events) + bytes((committed,))


def test_dbcop_parser_matches_commit_and_initial_transaction_rules(tmp_path: Path) -> None:
    """失败事务不进入历史，但其中成功 Event 的 Key 仍由初始事务初始化。"""

    payload = b"".join(
        (
            _i64(7),
            _i64(1),
            _i64(2),
            _i64(2),
            _i64(2),
            _string("PostgreSQL"),
            _string("start"),
            _string("end"),
            _i64(1),  # 实际 Session 数
            _i64(2),  # Session 中尝试两个事务
            _transaction([_event(True, 3, 9), _event(False, 4, 0, False)]),
            _transaction([_event(True, 8, 1)], committed=False),
        )
    )
    path = tmp_path / "history.bincode"
    path.write_bytes(payload)
    history = parse_dbcop_history(path)
    assert [tx.transaction_id for tx in history.transactions] == [1, 0]
    assert history.observed_keys == (3, 8)
    assert [event.key for event in history.transactions[-1].events] == [3, 8]


def _small_history() -> DbcopHistory:
    """T1、T2 同值写，T3 读取该值；T1->T2 是固定 SO。"""

    t1 = Transaction(1, 1, 0, (Event(10, 1, True),))
    t2 = Transaction(2, 1, 1, (Event(10, 1, True),))
    t3 = Transaction(3, 2, 0, (Event(10, 1, False),))
    initial = Transaction(0, 0, 0, (Event(10, 0, True),), is_initial=True)
    return DbcopHistory(
        header=DbcopHeader(0, 2, 1, 0, 0, "test", "", ""),
        sessions=(
            Session(1, (t1, t2)),
            Session(2, (t3,)),
            Session(0, (initial,)),
        ),
        observed_keys=(10,),
    )


def test_constraint_construction_and_fast_pruning() -> None:
    problem = construct_veristrong_problem(_small_history(), fast_prune=True)
    assert problem.pruning_consistent
    assert len(problem.all_ww_choices) == 3  # writer 集合为 {T0, T1, T2}
    assert len(problem.all_wr_choices) == 1
    assert problem.fixed_edges.has("SO", 1, 2)
    assert problem.fixed_edges.has("WW", 1, 2, 10)
    assert len(problem.remaining_ww_choices) == 2
    assert problem.remaining_wr_choices[0].candidate_writers == (1, 2)


def test_fast_pruning_conflict_is_certified_unsat() -> None:
    """T1->T2 的 SO 与唯一 writer 产生的 T2->T1 WR 直接冲突。"""

    t1 = Transaction(1, 1, 0, (Event(10, 1, False),))
    t2 = Transaction(2, 1, 1, (Event(10, 1, True),))
    initial = Transaction(0, 0, 0, (Event(10, 0, True),), is_initial=True)
    history = DbcopHistory(
        header=DbcopHeader(0, 1, 1, 2, 1, "test", "", ""),
        sessions=(Session(1, (t1, t2)), Session(0, (initial,))),
        observed_keys=(10,),
    )
    problem = construct_veristrong_problem(history, fast_prune=True)
    assert not problem.pruning_consistent
    result = decode_with_beam(problem, {}, {})
    assert result.status == DecodeStatus.CERTIFIED_UNSAT
    assert result.validation is None


def test_graph_model_grouped_loss_and_batch() -> None:
    problem = construct_veristrong_problem(_small_history(), fast_prune=True)
    graph = build_veristrong_graph(problem)
    assert graph.layout.transaction_count == 4
    assert graph.layout.key_count == 1
    assert graph.layout.decision_count == 4  # 2 个 WW + 2 个 WR candidate
    assert graph.layout.constraint_count == 3
    assert graph.ww_mask.sum().item() == 2
    assert graph.wr_mask.sum().item() == 2

    torch.manual_seed(3)
    model = VeriStrongDecisionNetwork(hidden_dim=8, processor_steps=1, dropout=0.0)
    outputs = model(graph)
    assert outputs.ww_logits.shape == (2,)
    assert outputs.wr_logits.shape == (2,)
    targets = VeriStrongTargets(
        ww_labels=torch.tensor([1.0, 0.0]),
        ww_labeled_mask=torch.tensor([True, True]),
        wr_feasible_mask=torch.tensor([True, False]),
        wr_labeled_group_mask=torch.tensor([True]),
    )
    losses = veristrong_decision_loss(outputs, targets)
    assert torch.isfinite(losses.total)
    assert losses.ww_examples == 2
    assert losses.wr_groups == 1
    losses.total.backward()

    mapped_targets = build_veristrong_targets(
        graph,
        ww_labels={graph.ww_choice_ids[0]: True},
        wr_feasible_writers={graph.wr_choice_ids[0]: {1, 2}},
    )
    assert mapped_targets.ww_labeled_mask.tolist() == [True, False]
    assert mapped_targets.wr_feasible_mask.tolist() == [True, True]

    normalizer = VeriStrongFeatureNormalizer.fit([graph])
    normalized = normalizer(graph)
    # one-hot 类型列不做标准化，连续列在单图训练集上的均值约为 0。
    assert torch.equal(normalized.decision_features[:, :2], graph.decision_features[:, :2])
    assert torch.allclose(normalized.decision_features[:, 4].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.equal(normalized.transaction_features[:, 4], graph.transaction_features[:, 4])

    batch = collate_veristrong_graphs([graph, graph])
    assert batch.graph_count == 2
    assert batch.layout.transaction_count == 8
    assert batch.wr_group_count == 2
    assert set(batch.decision_wr_group[batch.wr_mask].tolist()) == {0, 1}
    batch_outputs = model(batch)
    assert batch_outputs.ww_logits.shape == (4,)
    assert batch_outputs.wr_logits.shape == (4,)
    batch_targets = collate_veristrong_targets([targets, targets])
    batch_losses = veristrong_decision_loss(batch_outputs, batch_targets)
    assert batch_losses.ww_examples == 4
    assert batch_losses.wr_groups == 2


def test_witness_validation_and_beam_decoder() -> None:
    problem = construct_veristrong_problem(_small_history(), fast_prune=True)
    remaining = {choice.group_id: choice for choice in problem.remaining_ww_choices}
    pair_to_group = {
        (choice.left_transaction, choice.right_transaction): choice.group_id
        for choice in problem.remaining_ww_choices
    }
    # 0->1、1->2（固定）、2->0 构成 WW 环。
    cyclic = {
        pair_to_group[(0, 1)]: True,
        pair_to_group[(0, 2)]: False,
    }
    wr_group = problem.remaining_wr_choices[0].group_id
    assert not validate_witness(problem, cyclic, {wr_group: 1}).valid

    # 强烈偏好上述错误组合；beam=2 应保留另一条可认证分支。
    scores = {
        pair_to_group[(0, 1)]: 5.0,
        pair_to_group[(0, 2)]: -5.0,
    }
    result = decode_with_beam(
        problem,
        scores,
        {wr_group: {1: 0.0, 2: 0.0}},
        beam_size=2,
    )
    assert result.status == DecodeStatus.CERTIFIED_SAT
    assert result.validation is not None and result.validation.valid


def test_label_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    problem = construct_veristrong_problem(_small_history(), fast_prune=True)
    graph = build_veristrong_graph(problem)
    result = decode_with_beam(problem, {}, {}, beam_size=2)
    assert result.status == DecodeStatus.CERTIFIED_SAT

    labels = labels_from_witness(problem, result.ww_assignments, result.wr_assignments)
    label_path = tmp_path / "labels.json"
    save_veristrong_labels(labels, label_path)
    restored_labels = load_veristrong_labels(label_path, problem=problem)
    assert dict(restored_labels.ww_labels) == dict(labels.ww_labels)
    assert not restored_labels.is_veristrong_model_label
    with pytest.raises(ValueError, match="只接受 VeriStrong SAT model"):
        restored_labels.require_veristrong_model()
    assert restored_labels.to_targets(graph).ww_labeled_mask.all()

    torch.manual_seed(9)
    model = VeriStrongDecisionNetwork(hidden_dim=8, processor_steps=1, dropout=0.0)
    normalizer = VeriStrongFeatureNormalizer.fit([graph])
    model.eval()
    expected = model(normalizer(graph)).ww_logits
    checkpoint_path = tmp_path / "model.pt"
    save_veristrong_checkpoint(
        checkpoint_path, model, normalizer=normalizer, epoch=3, metadata={"说明": "测试"}
    )
    restored = load_veristrong_checkpoint(checkpoint_path)
    restored.model.eval()
    assert restored.normalizer is not None
    actual = restored.model(restored.normalizer(graph)).ww_logits
    assert restored.epoch == 3
    assert restored.metadata["说明"] == "测试"
    assert torch.equal(expected, actual)


def test_persistent_tensor_and_normalizer_cache(tmp_path: Path) -> None:
    """第二个 Dataset 应直接命中磁盘，标签内容变化后必须自动使用新缓存键。"""

    payload = b"".join(
        (
            _i64(7),
            _i64(1),
            _i64(2),
            _i64(2),
            _i64(2),
            _string("PostgreSQL"),
            _string("start"),
            _string("end"),
            _i64(2),
            _i64(2),
            _transaction([_event(True, 10, 1)]),
            _transaction([_event(True, 10, 1)]),
            _i64(1),
            _transaction([_event(False, 10, 1)]),
        )
    )
    history_path = tmp_path / "history.bincode"
    history_path.write_bytes(payload)
    problem = construct_veristrong_problem(parse_dbcop_history(history_path), fast_prune=True)
    witness = decode_with_beam(problem, {}, {}, beam_size=2)
    assert witness.status == DecodeStatus.CERTIFIED_SAT
    labels = labels_from_witness(
        problem,
        witness.ww_assignments,
        witness.wr_assignments,
        producer=LABEL_PRODUCER_VERISTRONG_MODEL,
    )
    label_path = tmp_path / "labels.json"
    save_veristrong_labels(labels, label_path)
    cache_dir = tmp_path / "cache"

    first = CachedLabeledVeriStrongHistoryDataset(
        [(history_path, label_path)],
        cache_dir,
        memory_cache=False,
    )
    first_sample = first[0]
    assert first.cache_hits == 0
    assert first.cache_misses == 1
    assert first.cache_path(0).exists()

    normalizer = VeriStrongFeatureNormalizer.fit([first_sample.graph])
    normalizer_path = save_cached_veristrong_normalizer(
        cache_dir,
        first.dataset_key,
        normalizer,
    )
    assert normalizer_path.exists()
    restored_normalizer = load_cached_veristrong_normalizer(cache_dir, first.dataset_key)
    assert restored_normalizer is not None
    for name, expected in normalizer.state_dict().items():
        assert torch.equal(expected, restored_normalizer.state_dict()[name])

    second = CachedLabeledVeriStrongHistoryDataset(
        [(history_path, label_path)],
        cache_dir,
        memory_cache=False,
    )
    second_sample = second[0]
    assert second.cache_hits == 1
    assert second.cache_misses == 0
    assert torch.equal(
        first_sample.graph.decision_features,
        second_sample.graph.decision_features,
    )
    assert torch.equal(first_sample.targets.ww_labels, second_sample.targets.ww_labels)

    rebuilt = CachedLabeledVeriStrongHistoryDataset(
        [(history_path, label_path)],
        cache_dir,
        memory_cache=False,
        rebuild=True,
    )
    rebuilt[0]
    rebuilt[0]
    assert rebuilt.cache_misses == 1
    assert rebuilt.cache_hits == 1

    # 即使 JSON 语义不变，只要标签文件内容变化也重新验证并生成独立缓存。
    label_path.write_text(label_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed = CachedLabeledVeriStrongHistoryDataset(
        [(history_path, label_path)],
        cache_dir,
        memory_cache=False,
    )
    assert changed.sample_keys != first.sample_keys
    changed[0]
    assert changed.cache_hits == 0
    assert changed.cache_misses == 1


def _veristrong_model_log(problem, ww_assignments, wr_assignments) -> str:
    """构造与本地 Acyclic-MiniSat Logger 完全同语义的测试输出。"""

    raw_to_dense = {
        transaction_id: dense
        for dense, transaction_id in enumerate(reversed(problem.transaction_ids))
    }
    lines = [
        "[trace] node map: "
        + ", ".join(
            f"({raw}, {dense})" for raw, dense in sorted(raw_to_dense.items())
        )
        + ",",
        "[Var to Theory Interpretion]",
    ]
    model_values: list[tuple[int, bool]] = []
    variable_id = 0
    for choice in problem.remaining_ww_choices:
        left_dense = raw_to_dense[choice.left_transaction]
        right_dense = raw_to_dense[choice.right_transaction]
        key_text = ", ".join(str(key) for key in choice.keys)
        lines.append(f"{variable_id}: WW, {left_dense} -> {right_dense}, keys = {key_text}")
        model_values.append((variable_id, bool(ww_assignments[choice.group_id])))
        variable_id += 1
        lines.append(f"{variable_id}: WW, {right_dense} -> {left_dense}, keys = {key_text}")
        model_values.append((variable_id, not bool(ww_assignments[choice.group_id])))
        variable_id += 1
    for choice in problem.remaining_wr_choices:
        reader_dense = raw_to_dense[choice.read_transaction]
        # 故意打乱候选顺序，证明标签映射不依赖 unordered_set 的输出顺序。
        for writer in reversed(choice.candidate_writers):
            writer_dense = raw_to_dense[writer]
            lines.append(
                f"{variable_id}: WR({choice.key}), {writer_dense} -> {reader_dense}"
            )
            model_values.append((variable_id, writer == wr_assignments[choice.group_id]))
            variable_id += 1

    # simplify 阶段可能打印同形赋值；解析器必须只采用 Model Founded 后的最终值。
    lines.extend(("[Model after simplify()]", "- 0 = false", "[Search Traits]"))
    lines.extend(("[Accept = true]", "[Model Founded]"))
    lines.extend(f"- {variable} = {str(value).lower()}" for variable, value in model_values)
    lines.append("[Eventual Accept = true]")
    return "\n".join(lines)


def test_veristrong_solver_model_is_authoritative_label_source() -> None:
    problem = construct_veristrong_problem(_small_history(), fast_prune=True)
    witness = decode_with_beam(problem, {}, {}, beam_size=2)
    assert witness.status == DecodeStatus.CERTIFIED_SAT
    model_text = _veristrong_model_log(
        problem, witness.ww_assignments, witness.wr_assignments
    )

    model = parse_veristrong_model_text(model_text, source="model.log")
    labels = labels_from_veristrong_model(problem, model)
    assert labels.producer == LABEL_PRODUCER_VERISTRONG_MODEL
    assert labels.source == "model.log"
    assert dict(labels.ww_labels) == dict(witness.ww_assignments)
    assert {
        group: next(iter(writers))
        for group, writers in labels.wr_feasible_writers.items()
    } == dict(witness.wr_assignments)
    labels.require_veristrong_model()


def test_veristrong_solver_model_rejects_incomplete_or_unsat_export() -> None:
    problem = construct_veristrong_problem(_small_history(), fast_prune=True)
    witness = decode_with_beam(problem, {}, {}, beam_size=2)
    model_text = _veristrong_model_log(
        problem, witness.ww_assignments, witness.wr_assignments
    )
    # 删除一个被选中的最终赋值后，该组不能再被静默标注。
    lines = model_text.splitlines()
    model_start = lines.index("[Model Founded]") + 1
    selected_index = next(
        index
        for index in range(model_start, len(lines))
        if lines[index].endswith("= true")
    )
    lines[selected_index] = "- 999 remains UNDEF"
    incomplete = "\n".join(lines)
    with pytest.raises(VeriStrongModelError, match="恰有一个 true"):
        labels_from_veristrong_model(problem, parse_veristrong_model_text(incomplete))

    unsat = parse_veristrong_model_text("[Accept = false]\n[Eventual Accept = false]")
    with pytest.raises(VeriStrongModelError, match="结果为 UNSAT"):
        labels_from_veristrong_model(problem, unsat)


def test_uploaded_fig10_history_golden_statistics() -> None:
    path = Path(
        "VeriStrong-main/history/fig10/"
        "20_10_15_5000_0.5_r_0.5_0.5_100/hist-00000/history.bincode"
    )
    if not path.exists():
        pytest.skip("未上传 VeriStrong Fig.10 历史")
    problem = construct_veristrong_problem(parse_dbcop_history(path), fast_prune=True)
    summary = problem_summary(problem)
    assert summary["transactions"] == 201
    assert summary["keys"] == 2030
    assert summary["ww_groups_total"] == 621
    assert summary["wr_groups_total"] == 1522
    assert summary["ww_groups_remaining"] == 128
    assert summary["wr_groups_remaining"] == 0
