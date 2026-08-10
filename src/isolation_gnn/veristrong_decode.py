"""神经分数到 VeriStrong assignment 的约束感知解码和 Witness 校验。"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .veristrong_data import DependencyEdges, VeriStrongProblem, WRChoice, WWChoice
from .veristrong_graph import VeriStrongGraph
from .veristrong_model import VeriStrongOutputs


class DecodeStatus(str, Enum):
    CERTIFIED_SAT = "CERTIFIED_SAT"
    CERTIFIED_UNSAT = "CERTIFIED_UNSAT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class WitnessValidation:
    valid: bool
    errors: tuple[str, ...]
    edges: DependencyEdges


@dataclass(frozen=True)
class DecodeResult:
    status: DecodeStatus
    ww_assignments: Mapping[int, bool]
    wr_assignments: Mapping[int, int]
    score: float
    expansions: int
    validation: WitnessValidation | None


def _is_acyclic(nodes: tuple[int, ...], edges: DependencyEdges) -> bool:
    adjacency = edges.adjacency(nodes)
    indegree = {node: 0 for node in adjacency}
    for targets in adjacency.values():
        for target in targets:
            indegree[target] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(indegree)


def _derive_all_rw(edges: DependencyEdges) -> None:
    """按照 WR(writer->reader) + WW(writer->next) 推导全部 RW。"""

    ww_by_writer_key: dict[tuple[int, int], set[int]] = {}
    for (writer, next_writer), keys in edges.by_type["WW"].items():
        for key in keys:
            ww_by_writer_key.setdefault((writer, key), set()).add(next_writer)
    for (writer, reader), wr_keys in list(edges.by_type["WR"].items()):
        for key in wr_keys:
            for next_writer in ww_by_writer_key.get((writer, key), ()):
                if reader != next_writer:
                    edges.add("RW", reader, next_writer, (key,))


def materialize_assignment(
    problem: VeriStrongProblem,
    ww_assignments: Mapping[int, bool],
    wr_assignments: Mapping[int, int],
    *,
    require_complete: bool = True,
) -> tuple[DependencyEdges, tuple[str, ...]]:
    """把部分或完整选择转换为依赖边，并执行确定性 RW 推导。"""

    errors: list[str] = []
    edges = problem.fixed_edges.copy()
    remaining_ww = {choice.group_id: choice for choice in problem.remaining_ww_choices}
    remaining_wr = {choice.group_id: choice for choice in problem.remaining_wr_choices}

    unknown_ww = set(ww_assignments) - set(remaining_ww)
    unknown_wr = set(wr_assignments) - set(remaining_wr)
    if unknown_ww:
        errors.append(f"包含未知 WW group：{sorted(unknown_ww)}")
    if unknown_wr:
        errors.append(f"包含未知 WR group：{sorted(unknown_wr)}")

    for group_id, choice in remaining_ww.items():
        if group_id not in ww_assignments:
            if require_complete:
                errors.append(f"WW group {group_id} 尚未赋值")
            continue
        left_to_right = bool(ww_assignments[group_id])
        source = choice.left_transaction if left_to_right else choice.right_transaction
        target = choice.right_transaction if left_to_right else choice.left_transaction
        edges.add("WW", source, target, choice.keys)

    for group_id, choice in remaining_wr.items():
        if group_id not in wr_assignments:
            if require_complete:
                errors.append(f"WR group {group_id} 尚未赋值")
            continue
        writer = int(wr_assignments[group_id])
        if writer not in choice.candidate_writers:
            errors.append(f"WR group {group_id} 的 writer {writer} 不在候选集合中")
            continue
        edges.add("WR", writer, choice.read_transaction, (choice.key,))

    _derive_all_rw(edges)
    return edges, tuple(errors)


def validate_witness(
    problem: VeriStrongProblem,
    ww_assignments: Mapping[int, bool],
    wr_assignments: Mapping[int, int],
) -> WitnessValidation:
    """独立验证选择完整性、WW/WR 语义、RW 推导和最终无环性。"""

    edges, materialization_errors = materialize_assignment(
        problem, ww_assignments, wr_assignments, require_complete=True
    )
    errors = list(materialization_errors)

    # 每个原始 WW 事务对、每个共享 Key 必须恰好存在一个方向。
    for choice in problem.all_ww_choices:
        for key in choice.keys:
            forward = edges.has(
                "WW", choice.left_transaction, choice.right_transaction, key
            )
            backward = edges.has(
                "WW", choice.right_transaction, choice.left_transaction, key
            )
            if forward == backward:
                errors.append(
                    f"WW group {choice.group_id} 在 Key {key} 上不是恰好一个方向"
                )

    # 每个读取约束必须且只能选择一个原始候选 writer。
    for choice in problem.all_wr_choices:
        selected = [
            writer
            for writer in choice.candidate_writers
            if edges.has("WR", writer, choice.read_transaction, choice.key)
        ]
        if len(selected) != 1:
            errors.append(
                f"WR group {choice.group_id} 选择了 {len(selected)} 个候选，而不是 1 个"
            )

    if not _is_acyclic(problem.transaction_ids, edges):
        errors.append("SO ∪ WR ∪ WW ∪ RW 包含环")
    return WitnessValidation(valid=not errors, errors=tuple(errors), edges=edges)


def scores_from_outputs(
    graph: VeriStrongGraph,
    outputs: VeriStrongOutputs,
) -> tuple[dict[int, float], dict[int, dict[int, float]]]:
    """把模型的紧凑 Tensor 输出还原为原始 group/transaction ID 分数。"""

    if graph.graph_count != 1:
        raise ValueError("批处理图的原始 ID 可能重复，请拆分输出后逐图解码")
    if outputs.ww_logits.numel() != len(graph.ww_choice_ids):
        raise ValueError("WW logits 数量与图中的 WW choice 数量不一致")
    ww_scores = {
        group_id: float(logit)
        for group_id, logit in zip(graph.ww_choice_ids, outputs.ww_logits.detach().cpu().tolist())
    }

    wr_scores: dict[int, dict[int, float]] = {group_id: {} for group_id in graph.wr_choice_ids}
    logits = outputs.wr_logits.detach().cpu().tolist()
    groups = outputs.wr_group_indices.detach().cpu().tolist()
    writers = outputs.wr_candidate_transaction_indices.detach().cpu().tolist()
    for logit, dense_group, writer_local in zip(logits, groups, writers):
        raw_group = graph.wr_choice_ids[dense_group]
        raw_writer = graph.transaction_ids[writer_local]
        wr_scores[raw_group][raw_writer] = float(logit)
    return ww_scores, wr_scores


@dataclass
class _BeamState:
    score: float
    ww: dict[int, bool]
    wr: dict[int, int]
    # 每个整数的 bit 位表示对应事务的可达集合，包含自身。
    reachability: tuple[int, ...]
    # 只记录当前分支新增的 WW/WR。固定边放在共享只读索引中，避免每次扩展复制整张图。
    selected_ww: dict[tuple[int, int], frozenset[int]]
    selected_wr: dict[tuple[int, int], frozenset[int]]


def _index_dependencies(
    edges: DependencyEdges,
) -> tuple[
    dict[tuple[int, int], frozenset[int]],
    dict[tuple[int, int], frozenset[int]],
]:
    """把依赖边索引为 ``(writer, key) -> 对端事务``。"""

    ww: dict[tuple[int, int], set[int]] = {}
    wr: dict[tuple[int, int], set[int]] = {}
    for (writer, next_writer), keys in edges.by_type["WW"].items():
        for key in keys:
            ww.setdefault((writer, key), set()).add(next_writer)
    for (writer, reader), keys in edges.by_type["WR"].items():
        for key in keys:
            wr.setdefault((writer, key), set()).add(reader)
    return (
        {key: frozenset(values) for key, values in ww.items()},
        {key: frozenset(values) for key, values in wr.items()},
    )


def _reachability_bits(
    nodes: tuple[int, ...], adjacency: Mapping[int, set[int]]
) -> tuple[tuple[int, ...], dict[int, int]]:
    """按 DAG 逆拓扑序计算 bitset 可达闭包。"""

    transaction_index = {transaction_id: index for index, transaction_id in enumerate(nodes)}
    indegree = {node: 0 for node in nodes}
    for targets in adjacency.values():
        for target in targets:
            indegree[target] += 1
    queue = deque(node for node in nodes if indegree[node] == 0)
    topological: list[int] = []
    while queue:
        node = queue.popleft()
        topological.append(node)
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(topological) != len(nodes):
        raise ValueError("固定依赖图包含环，不能构造可达闭包")

    closure = [1 << index for index in range(len(nodes))]
    for node in reversed(topological):
        index = transaction_index[node]
        for target in adjacency[node]:
            closure[index] |= closure[transaction_index[target]]
    return tuple(closure), transaction_index


def _add_acyclic_edge(
    reachability: list[int],
    transaction_index: Mapping[int, int],
    source: int,
    target: int,
) -> bool:
    """用 bitset 增量更新传递闭包；若会成环则返回 ``False``。"""

    source_index = transaction_index[source]
    target_index = transaction_index[target]
    source_bit = 1 << source_index
    target_bit = 1 << target_index
    if reachability[target_index] & source_bit:
        return False
    if reachability[source_index] & target_bit:
        return True
    descendants = reachability[target_index]
    # 所有能到达 source 的祖先现在都能到达 target 及其后继。
    for index, reached in enumerate(reachability):
        if index == source_index or reached & source_bit:
            reachability[index] = reached | descendants
    return True


def _extend_state(
    state: _BeamState,
    *,
    kind: str,
    choice: WWChoice | WRChoice,
    option: object,
    log_probability: float,
    fixed_ww: Mapping[tuple[int, int], frozenset[int]],
    fixed_wr: Mapping[tuple[int, int], frozenset[int]],
    transaction_index: Mapping[int, int],
) -> _BeamState | None:
    """只传播本次选择新增的依赖；出现环时立即剪掉该分支。"""

    reachability = list(state.reachability)
    selected_ww = state.selected_ww
    selected_wr = state.selected_wr
    ww = dict(state.ww)
    wr = dict(state.wr)

    if kind == "ww":
        assert isinstance(choice, WWChoice)
        left_to_right = bool(option)
        source = choice.left_transaction if left_to_right else choice.right_transaction
        target = choice.right_transaction if left_to_right else choice.left_transaction
        if not _add_acyclic_edge(reachability, transaction_index, source, target):
            return None

        selected_ww = dict(selected_ww)
        for key in choice.keys:
            # WR(source -> reader) 与新 WW(source -> target) 推出 RW(reader -> target)。
            readers = fixed_wr.get((source, key), frozenset()) | selected_wr.get(
                (source, key), frozenset()
            )
            for reader in readers:
                if reader == target:
                    continue
                if not _add_acyclic_edge(
                    reachability, transaction_index, reader, target
                ):
                    return None
            index = (source, key)
            selected_ww[index] = selected_ww.get(index, frozenset()) | {target}
        ww[choice.group_id] = left_to_right
    else:
        assert isinstance(choice, WRChoice)
        writer = int(option)
        reader = choice.read_transaction
        if writer not in choice.candidate_writers:
            return None
        if not _add_acyclic_edge(reachability, transaction_index, writer, reader):
            return None

        # 新 WR(writer -> reader) 与已有 WW(writer -> next) 推出 RW(reader -> next)。
        next_writers = fixed_ww.get((writer, choice.key), frozenset()) | selected_ww.get(
            (writer, choice.key), frozenset()
        )
        for next_writer in next_writers:
            if next_writer == reader:
                continue
            if not _add_acyclic_edge(
                reachability, transaction_index, reader, next_writer
            ):
                return None
        selected_wr = dict(selected_wr)
        index = (writer, choice.key)
        selected_wr[index] = selected_wr.get(index, frozenset()) | {reader}
        wr[choice.group_id] = writer

    return _BeamState(
        score=state.score + log_probability,
        ww=ww,
        wr=wr,
        reachability=tuple(reachability),
        selected_ww=selected_ww,
        selected_wr=selected_wr,
    )


def _log_sigmoid(value: float) -> float:
    return -math.log1p(math.exp(-value)) if value >= 0 else value - math.log1p(math.exp(value))


def _choice_options(
    problem: VeriStrongProblem,
    ww_scores: Mapping[int, float],
    wr_scores: Mapping[int, Mapping[int, float]],
) -> list[tuple[float, str, WWChoice | WRChoice, list[tuple[object, float]]]]:
    """构造按模型置信度排序的选择组及其归一化对数概率。"""

    groups: list[tuple[float, str, WWChoice | WRChoice, list[tuple[object, float]]]] = []
    for choice in problem.remaining_ww_choices:
        logit = float(ww_scores.get(choice.group_id, 0.0))
        options: list[tuple[object, float]] = [
            (True, _log_sigmoid(logit)),
            (False, _log_sigmoid(-logit)),
        ]
        options.sort(key=lambda item: item[1], reverse=True)
        groups.append((math.exp(options[0][1]), "ww", choice, options))

    for choice in problem.remaining_wr_choices:
        raw_scores = wr_scores.get(choice.group_id, {})
        scores = [float(raw_scores.get(writer, 0.0)) for writer in choice.candidate_writers]
        maximum = max(scores)
        denominator = maximum + math.log(sum(math.exp(score - maximum) for score in scores))
        options = [
            (writer, score - denominator)
            for writer, score in zip(choice.candidate_writers, scores)
        ]
        options.sort(key=lambda item: item[1], reverse=True)
        groups.append((math.exp(options[0][1]), "wr", choice, options))

    # 高置信度组优先，使贪心快速路径尽早固定模型最确定的选择。
    groups.sort(key=lambda item: item[0], reverse=True)
    return groups


def decode_with_beam(
    problem: VeriStrongProblem,
    ww_scores: Mapping[int, float],
    wr_scores: Mapping[int, Mapping[int, float]],
    *,
    beam_size: int = 1,
    max_expansions: int = 100_000,
) -> DecodeResult:
    """使用模型分数、增量环检查和有限 Beam Search 构造可认证 Witness。"""

    if beam_size < 1 or max_expansions < 1:
        raise ValueError("beam_size 和 max_expansions 必须为正数")
    if not problem.pruning_consistent:
        # 两个方向/全部 writer 均被已知可达关系排除，是确定性冲突证明。
        return DecodeResult(DecodeStatus.CERTIFIED_UNSAT, {}, {}, -math.inf, 0, None)

    # 剪枝器通常已经推导固定 RW；这里再闭包一次，确保自定义 Problem 也满足同一语义。
    fixed_edges = problem.fixed_edges.copy()
    _derive_all_rw(fixed_edges)
    if not _is_acyclic(problem.transaction_ids, fixed_edges):
        return DecodeResult(DecodeStatus.UNKNOWN, {}, {}, -math.inf, 0, None)
    fixed_ww, fixed_wr = _index_dependencies(fixed_edges)
    reachability, transaction_index = _reachability_bits(
        problem.transaction_ids, fixed_edges.adjacency(problem.transaction_ids)
    )
    states = [
        _BeamState(
            score=0.0,
            ww={},
            wr={},
            reachability=reachability,
            selected_ww={},
            selected_wr={},
        )
    ]
    expansions = 0
    for _, kind, choice, options in _choice_options(problem, ww_scores, wr_scores):
        next_states: list[_BeamState] = []
        for state in states:
            for option, log_probability in options:
                if expansions >= max_expansions:
                    return DecodeResult(
                        DecodeStatus.UNKNOWN,
                        state.ww,
                        state.wr,
                        state.score,
                        expansions,
                        None,
                    )
                expansions += 1
                extended = _extend_state(
                    state,
                    kind=kind,
                    choice=choice,
                    option=option,
                    log_probability=log_probability,
                    fixed_ww=fixed_ww,
                    fixed_wr=fixed_wr,
                    transaction_index=transaction_index,
                )
                if extended is not None:
                    next_states.append(extended)
        if not next_states:
            return DecodeResult(DecodeStatus.UNKNOWN, {}, {}, -math.inf, expansions, None)
        next_states.sort(key=lambda state: state.score, reverse=True)
        states = next_states[:beam_size]

    for state in states:
        validation = validate_witness(problem, state.ww, state.wr)
        if validation.valid:
            return DecodeResult(
                DecodeStatus.CERTIFIED_SAT,
                state.ww,
                state.wr,
                state.score,
                expansions,
                validation,
            )
    best = states[0]
    return DecodeResult(
        DecodeStatus.UNKNOWN,
        best.ww,
        best.wr,
        best.score,
        expansions,
        validate_witness(problem, best.ww, best.wr),
    )


__all__ = [
    "DecodeResult",
    "DecodeStatus",
    "WitnessValidation",
    "decode_with_beam",
    "materialize_assignment",
    "scores_from_outputs",
    "validate_witness",
]
