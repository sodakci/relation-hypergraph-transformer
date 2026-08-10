"""解析 VeriStrong Acyclic-MiniSat 模型，并转换为可复验的监督标签。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .veristrong_data import VeriStrongProblem
from .veristrong_labels import (
    LABEL_PRODUCER_VERISTRONG_MODEL,
    VeriStrongLabels,
    labels_from_witness,
)


class VeriStrongModelError(ValueError):
    """VeriStrong 模型缺失、格式错误，或无法与当前问题一一对应。"""


@dataclass(frozen=True)
class SolverTheoryVariable:
    """一个 MiniSat 变量对应的 VeriStrong 理论边。"""

    variable_id: int
    kind: str
    dense_source: int
    dense_target: int
    keys: tuple[int, ...]


@dataclass(frozen=True)
class VeriStrongSolverModel:
    """从一次 VeriStrong 求解日志恢复出的 SAT model。"""

    accepted: bool
    dense_to_raw_transaction: Mapping[int, int]
    theory_variables: Mapping[int, SolverTheoryVariable]
    assignments: Mapping[int, bool | None]
    source: str | None = None

    def __post_init__(self) -> None:
        # 使用只读副本，防止调用方在标签映射期间篡改模型内容。
        object.__setattr__(
            self,
            "dense_to_raw_transaction",
            MappingProxyType(
                {
                    int(dense): int(raw)
                    for dense, raw in self.dense_to_raw_transaction.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "theory_variables",
            MappingProxyType(
                {int(variable): value for variable, value in self.theory_variables.items()}
            ),
        )
        object.__setattr__(
            self,
            "assignments",
            MappingProxyType(
                {
                    int(variable): None if value is None else bool(value)
                    for variable, value in self.assignments.items()
                }
            ),
        )


_NODE_PAIR = re.compile(r"\((-?\d+)\s*,\s*(\d+)\)")
_WW_VARIABLE = re.compile(
    r"(?:^|\s)(\d+)\s*:\s*WW\s*,\s*(-?\d+)\s*->\s*(-?\d+)"
    r"\s*,\s*keys\s*=\s*(.*)\s*$"
)
_WR_VARIABLE = re.compile(
    r"(?:^|\s)(\d+)\s*:\s*WR\((-?\d+)\)\s*,\s*(-?\d+)\s*->\s*(-?\d+)"
)
_MODEL_VALUE = re.compile(r"^\s*-\s*(\d+)\s*=\s*(true|false)\s*$", re.IGNORECASE)
_MODEL_UNDEF = re.compile(r"^\s*-\s*(\d+)\s+remains\s+UNDEF\s*$", re.IGNORECASE)
_EVENTUAL_ACCEPT = re.compile(
    r"\[\s*Eventual\s+Accept\s*=\s*(true|false)\s*\]", re.IGNORECASE
)
_ACCEPT = re.compile(r"\[\s*Accept\s*=\s*(true|false)\s*\]", re.IGNORECASE)
_PLAIN_ACCEPT = re.compile(r"^\s*accept\s*:\s*(true|false)\s*$", re.IGNORECASE | re.MULTILINE)


def _last_marker(text: str, marker: str, *, before: int | None = None) -> int:
    """寻找指定边界前最后一个标记，失败时给出可操作的错误。"""

    position = text.rfind(marker, 0, before)
    if position < 0:
        raise VeriStrongModelError(f"模型导出中缺少 {marker} 段")
    return position


def _parse_accept(text: str) -> bool:
    """优先采用最外层 Eventual Accept，避免误读 simplify 中间结果。"""

    eventual = _EVENTUAL_ACCEPT.findall(text)
    if eventual:
        return eventual[-1].lower() == "true"
    accepts = _ACCEPT.findall(text)
    if accepts:
        return accepts[-1].lower() == "true"
    plain_accepts = _PLAIN_ACCEPT.findall(text)
    if plain_accepts:
        return plain_accepts[-1].lower() == "true"
    raise VeriStrongModelError("模型导出中没有 Accept/Eventual Accept 结果")


def _parse_node_map(text: str, *, before: int) -> dict[int, int]:
    """读取最后一次求解使用的 raw transaction -> dense ID 映射。"""

    marker = text.rfind("node map:", 0, before)
    # node map 走 Boost.Log，而 MiniSat model 走 stdout；用户若合并两个流，
    # 缓冲可能让 node map 落在 model 后。单次导出时允许从全文兜底读取。
    if marker < 0:
        marker = text.rfind("node map:")
    if marker < 0:
        return {}
    line_end = text.find("\n", marker)
    line = text[marker:] if line_end < 0 else text[marker:line_end]
    dense_to_raw: dict[int, int] = {}
    raw_to_dense: dict[int, int] = {}
    for raw_text, dense_text in _NODE_PAIR.findall(line):
        raw = int(raw_text)
        dense = int(dense_text)
        if dense in dense_to_raw and dense_to_raw[dense] != raw:
            raise VeriStrongModelError(f"dense transaction ID {dense} 对应多个原始事务")
        if raw in raw_to_dense and raw_to_dense[raw] != dense:
            raise VeriStrongModelError(f"原始事务 {raw} 对应多个 dense transaction ID")
        dense_to_raw[dense] = raw
        raw_to_dense[raw] = dense
    if not dense_to_raw:
        raise VeriStrongModelError("找到了 node map，但没有解析出任何事务 ID 对")
    return dense_to_raw


def _parse_theory_variables(segment: str) -> dict[int, SolverTheoryVariable]:
    """解析 WW/WR 变量；RW 辅助变量不属于监督目标，故有意忽略。"""

    variables: dict[int, SolverTheoryVariable] = {}
    for line in segment.splitlines():
        ww_match = _WW_VARIABLE.search(line)
        if ww_match is not None:
            variable_id = int(ww_match.group(1))
            key_text = ww_match.group(4)
            keys = tuple(sorted(int(value) for value in re.findall(r"-?\d+", key_text)))
            if not keys:
                raise VeriStrongModelError(f"WW 变量 {variable_id} 没有 key")
            variable = SolverTheoryVariable(
                variable_id=variable_id,
                kind="WW",
                dense_source=int(ww_match.group(2)),
                dense_target=int(ww_match.group(3)),
                keys=keys,
            )
            if variable_id in variables and variables[variable_id] != variable:
                raise VeriStrongModelError(f"变量 {variable_id} 出现冲突的 WW 定义")
            variables[variable_id] = variable
            continue

        wr_match = _WR_VARIABLE.search(line)
        if wr_match is not None:
            variable_id = int(wr_match.group(1))
            variable = SolverTheoryVariable(
                variable_id=variable_id,
                kind="WR",
                dense_source=int(wr_match.group(3)),
                dense_target=int(wr_match.group(4)),
                keys=(int(wr_match.group(2)),),
            )
            if variable_id in variables and variables[variable_id] != variable:
                raise VeriStrongModelError(f"变量 {variable_id} 出现冲突的 WR 定义")
            variables[variable_id] = variable
    return variables


def _parse_model_assignments(segment: str) -> dict[int, bool | None]:
    """只读取 `[Model Founded]` 后的最终 model，忽略 simplify 阶段赋值。"""

    assignments: dict[int, bool | None] = {}
    for line in segment.splitlines():
        value_match = _MODEL_VALUE.match(line)
        if value_match is not None:
            variable_id = int(value_match.group(1))
            value = value_match.group(2).lower() == "true"
        else:
            undef_match = _MODEL_UNDEF.match(line)
            if undef_match is None:
                continue
            variable_id = int(undef_match.group(1))
            value = None
        if variable_id in assignments and assignments[variable_id] != value:
            raise VeriStrongModelError(f"最终 model 中变量 {variable_id} 有多个不同赋值")
        assignments[variable_id] = value
    return assignments


def parse_veristrong_model_text(
    text: str,
    *,
    source: str | None = None,
) -> VeriStrongSolverModel:
    """解析启用 Logger 后的 VeriStrong Acyclic-MiniSat 完整输出。"""

    accepted = _parse_accept(text)
    model_marker = text.rfind("[Model Founded]")

    # UNSAT 没有 SAT model，仍返回 accepted=False，供批处理程序正确跳过。
    if not accepted:
        return VeriStrongSolverModel(False, {}, {}, {}, source=source)

    # 全部约束在 pruning 中确定时 n_vars==0，VeriStrong 会直接返回 true，
    # 此时没有 Model Founded 段，转换阶段仅允许目标问题也没有剩余选择。
    if model_marker < 0:
        return VeriStrongSolverModel(True, {}, {}, {}, source=source)

    theory_marker = _last_marker(text, "[Var to Theory Interpretion]", before=model_marker)
    dense_to_raw = _parse_node_map(text, before=theory_marker)
    variables = _parse_theory_variables(text[theory_marker:model_marker])
    assignments = _parse_model_assignments(text[model_marker + len("[Model Founded]") :])
    if variables and not assignments:
        raise VeriStrongModelError("Model Founded 段没有解析出任何变量赋值")
    return VeriStrongSolverModel(
        accepted=True,
        dense_to_raw_transaction=dense_to_raw,
        theory_variables=variables,
        assignments=assignments,
        source=source,
    )


def load_veristrong_solver_model(path: str | Path) -> VeriStrongSolverModel:
    """从 VeriStrong 求解输出文件读取 model。"""

    source = Path(path)
    return parse_veristrong_model_text(
        source.read_text(encoding="utf-8", errors="replace"),
        source=str(source.resolve()),
    )


def _raw_transaction(model: VeriStrongSolverModel, dense_id: int) -> int:
    try:
        return model.dense_to_raw_transaction[dense_id]
    except KeyError as exc:
        raise VeriStrongModelError(f"node map 缺少 dense transaction ID {dense_id}") from exc


def labels_from_veristrong_model(
    problem: VeriStrongProblem,
    model: VeriStrongSolverModel,
) -> VeriStrongLabels:
    """把 VeriStrong SAT model 映射为稳定 group ID 标签，并独立复验 witness。"""

    if not model.accepted:
        raise VeriStrongModelError("VeriStrong 结果为 UNSAT，不能生成 SAT 监督标签")
    if not problem.pruning_consistent:
        raise VeriStrongModelError("当前问题在 fast pruning 阶段冲突，不能绑定 SAT model")

    expected_choice_count = len(problem.remaining_ww_choices) + len(
        problem.remaining_wr_choices
    )
    if expected_choice_count == 0:
        if any(
            variable.kind in {"WW", "WR"}
            for variable in model.theory_variables.values()
        ):
            raise VeriStrongModelError("Python 问题已无剩余选择，但 VeriStrong model 仍含 WW/WR 变量")
        return labels_from_witness(
            problem,
            {},
            {},
            source=model.source,
            producer=LABEL_PRODUCER_VERISTRONG_MODEL,
        )

    if not model.dense_to_raw_transaction:
        raise VeriStrongModelError(
            "model 缺少 node map，无法把 MiniSat dense ID 还原为原始事务 ID；"
            "请使用启用了 model-only 导出的 checker"
        )

    ww_lookup = {
        (
            frozenset((choice.left_transaction, choice.right_transaction)),
            frozenset(choice.keys),
        ): choice
        for choice in problem.remaining_ww_choices
    }
    wr_lookup = {
        (choice.read_transaction, choice.key): choice
        for choice in problem.remaining_wr_choices
    }
    ww_variables: dict[int, list[tuple[int, int, int]]] = {
        choice.group_id: [] for choice in problem.remaining_ww_choices
    }
    wr_variables: dict[int, list[tuple[int, int]]] = {
        choice.group_id: [] for choice in problem.remaining_wr_choices
    }

    for variable_id, variable in model.theory_variables.items():
        if variable.kind == "WW":
            source = _raw_transaction(model, variable.dense_source)
            target = _raw_transaction(model, variable.dense_target)
            choice = ww_lookup.get(
                (frozenset((source, target)), frozenset(variable.keys))
            )
            if choice is None:
                raise VeriStrongModelError(
                    f"VeriStrong WW 变量 {variable_id} ({source}->{target}, keys={variable.keys}) "
                    "无法匹配 Python 剪枝后的 WW group"
                )
            ww_variables[choice.group_id].append((variable_id, source, target))
        elif variable.kind == "WR":
            writer = _raw_transaction(model, variable.dense_source)
            reader = _raw_transaction(model, variable.dense_target)
            key = variable.keys[0]
            choice = wr_lookup.get((reader, key))
            if choice is None or writer not in choice.candidate_writers:
                raise VeriStrongModelError(
                    f"VeriStrong WR 变量 {variable_id} ({writer}->{reader}, key={key}) "
                    "无法匹配 Python 剪枝后的 WR candidate"
                )
            wr_variables[choice.group_id].append((variable_id, writer))

    ww_assignments: dict[int, bool] = {}
    for choice in problem.remaining_ww_choices:
        variables = ww_variables[choice.group_id]
        orientations = {(source, target) for _, source, target in variables}
        expected_orientations = {
            (choice.left_transaction, choice.right_transaction),
            (choice.right_transaction, choice.left_transaction),
        }
        if len(variables) != 2 or orientations != expected_orientations:
            raise VeriStrongModelError(
                f"WW group {choice.group_id} 应有两个相反方向变量，实际为 {variables}"
            )
        selected = [item for item in variables if model.assignments.get(item[0]) is True]
        if len(selected) != 1:
            raise VeriStrongModelError(
                f"WW group {choice.group_id} 在最终 model 中应恰有一个 true，实际为 {selected}"
            )
        _, source, target = selected[0]
        ww_assignments[choice.group_id] = (
            source == choice.left_transaction and target == choice.right_transaction
        )

    wr_assignments: dict[int, int] = {}
    for choice in problem.remaining_wr_choices:
        variables = wr_variables[choice.group_id]
        actual_writers = [writer for _, writer in variables]
        if len(variables) != len(choice.candidate_writers) or set(actual_writers) != set(
            choice.candidate_writers
        ):
            raise VeriStrongModelError(
                f"WR group {choice.group_id} 的 model writer 集合 {sorted(actual_writers)} "
                f"与候选集合 {list(choice.candidate_writers)} 不一致"
            )
        selected = [writer for variable, writer in variables if model.assignments.get(variable) is True]
        if len(selected) != 1:
            raise VeriStrongModelError(
                f"WR group {choice.group_id} 在最终 model 中应恰有一个 true，实际为 {selected}"
            )
        wr_assignments[choice.group_id] = selected[0]

    # labels_from_witness 会重新物化所有 WW/WR/RW 边并检查最终图无环。
    try:
        return labels_from_witness(
            problem,
            ww_assignments,
            wr_assignments,
            source=model.source,
            producer=LABEL_PRODUCER_VERISTRONG_MODEL,
        )
    except (KeyError, ValueError) as exc:
        raise VeriStrongModelError(f"VeriStrong model 未通过独立 witness 校验：{exc}") from exc


__all__ = [
    "SolverTheoryVariable",
    "VeriStrongModelError",
    "VeriStrongSolverModel",
    "labels_from_veristrong_model",
    "load_veristrong_solver_model",
    "parse_veristrong_model_text",
]
