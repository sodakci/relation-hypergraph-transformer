"""VeriStrong DBCop 历史解析、约束构造和确定性快速剪枝。"""

from __future__ import annotations

import json
import struct
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping, Sequence


class HistoryFormatError(ValueError):
    """历史二进制内容不完整或不符合 DBCop 格式。"""


class ConstraintConstructionError(ValueError):
    """历史无法构造合法的 VeriStrong 约束。"""


@dataclass(frozen=True)
class DbcopHeader:
    history_id: int
    declared_sessions: int
    declared_keys: int
    declared_transactions_per_session: int
    declared_events_per_transaction: int
    database: str
    start_time: str
    end_time: str


@dataclass(frozen=True)
class Event:
    key: int
    value: int
    is_write: bool


@dataclass(frozen=True)
class Transaction:
    transaction_id: int
    session_id: int
    session_position: int
    events: tuple[Event, ...]
    is_initial: bool = False


@dataclass(frozen=True)
class Session:
    session_id: int
    transactions: tuple[Transaction, ...]


@dataclass(frozen=True)
class DbcopHistory:
    """已经应用 VeriStrong 提交/成功过滤规则的历史。"""

    header: DbcopHeader
    sessions: tuple[Session, ...]
    observed_keys: tuple[int, ...]
    source: str | None = None

    @property
    def transactions(self) -> tuple[Transaction, ...]:
        return tuple(tx for session in self.sessions for tx in session.transactions)

    @property
    def transaction_by_id(self) -> dict[int, Transaction]:
        return {tx.transaction_id: tx for tx in self.transactions}


@dataclass(frozen=True)
class TransactionAccess:
    """与 constraints_of() 一致的第一次外部读和最后一次写。"""

    reads: Mapping[int, int]
    writes: Mapping[int, int]


@dataclass(frozen=True)
class WWChoice:
    group_id: int
    left_transaction: int
    right_transaction: int
    keys: tuple[int, ...]


@dataclass(frozen=True)
class WRChoice:
    group_id: int
    key: int
    value: int
    read_transaction: int
    candidate_writers: tuple[int, ...]


@dataclass
class DependencyEdges:
    """按类型保存依赖边及其 Key 集合。"""

    by_type: dict[str, dict[tuple[int, int], set[int]]] = field(
        default_factory=lambda: {name: {} for name in ("SO", "WR", "WW", "RW")}
    )

    def copy(self) -> "DependencyEdges":
        return DependencyEdges(
            {
                kind: {edge: set(keys) for edge, keys in edges.items()}
                for kind, edges in self.by_type.items()
            }
        )

    def add(self, kind: str, source: int, target: int, keys: Iterable[int] = ()) -> None:
        if kind not in self.by_type:
            raise KeyError(f"未知依赖边类型：{kind}")
        if source == target:
            raise ConstraintConstructionError(f"不允许自环：{kind} {source}->{target}")
        self.by_type[kind].setdefault((source, target), set()).update(keys)

    def has(self, kind: str, source: int, target: int, key: int | None = None) -> bool:
        keys = self.by_type[kind].get((source, target))
        return keys is not None and (key is None or key in keys)

    def typed_edges(self) -> Iterable[tuple[str, int, int, frozenset[int]]]:
        for kind in ("SO", "WR", "WW", "RW"):
            for (source, target), keys in sorted(self.by_type[kind].items()):
                yield kind, source, target, frozenset(keys)

    def adjacency(self, nodes: Iterable[int]) -> dict[int, set[int]]:
        result = {node: set() for node in nodes}
        for edges in self.by_type.values():
            for source, target in edges:
                result[source].add(target)
                result.setdefault(target, set())
        return result


@dataclass(frozen=True)
class VeriStrongProblem:
    """一个可供神经模型和约束解码器共同使用的问题实例。"""

    history: DbcopHistory
    accesses: Mapping[int, TransactionAccess]
    all_ww_choices: tuple[WWChoice, ...]
    all_wr_choices: tuple[WRChoice, ...]
    fixed_edges: DependencyEdges
    remaining_ww_choices: tuple[WWChoice, ...]
    remaining_wr_choices: tuple[WRChoice, ...]
    pruning_consistent: bool = True
    pruning_rounds: int = 0

    @property
    def transaction_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.accesses))


class _BinaryReader:
    """带偏移量和 EOF 检查的 DBCop 小端读取器。"""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.offset = 0

    def _read_exact(self, size: int) -> bytes:
        data = self.stream.read(size)
        if len(data) != size:
            raise HistoryFormatError(
                f"历史在偏移 {self.offset} 处提前结束：需要 {size} 字节，只读到 {len(data)} 字节"
            )
        self.offset += size
        return data

    def int64(self) -> int:
        return struct.unpack("<q", self._read_exact(8))[0]

    def boolean(self) -> bool:
        value = self._read_exact(1)[0]
        if value not in (0, 1):
            raise HistoryFormatError(f"偏移 {self.offset - 1} 处的 bool 值非法：{value}")
        return bool(value)

    def string(self) -> str:
        size = self.int64()
        if size < 0 or size > 16 * 1024 * 1024:
            raise HistoryFormatError(f"非法字符串长度：{size}")
        try:
            return self._read_exact(size).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HistoryFormatError("历史头部字符串不是合法 UTF-8") from exc


def _checked_count(name: str, value: int, maximum: int = 100_000_000) -> int:
    if value < 0 or value > maximum:
        raise HistoryFormatError(f"{name} 数量非法：{value}")
    return value


def parse_dbcop_history(path: str | Path) -> DbcopHistory:
    """按照 VeriStrong ``parse_dbcop_history`` 的规则读取历史文件。"""

    source = Path(path)
    with source.open("rb") as stream:
        reader = _BinaryReader(stream)
        header = DbcopHeader(
            history_id=reader.int64(),
            declared_sessions=reader.int64(),
            declared_keys=reader.int64(),
            declared_transactions_per_session=reader.int64(),
            declared_events_per_transaction=reader.int64(),
            database=reader.string(),
            start_time=reader.string(),
            end_time=reader.string(),
        )

        sessions: list[Session] = []
        observed_keys: set[int] = set()
        next_transaction_id = 1
        session_count = _checked_count("Session", reader.int64())
        for session_id in range(1, session_count + 1):
            committed: list[Transaction] = []
            transaction_count = _checked_count("Transaction", reader.int64())
            for _ in range(transaction_count):
                transaction_id = next_transaction_id
                next_transaction_id += 1
                events: list[Event] = []
                event_count = _checked_count("Event", reader.int64())
                for _ in range(event_count):
                    is_write = reader.boolean()
                    key = reader.int64()
                    value = reader.int64()
                    success = reader.boolean()
                    if success:
                        # C++ 解析器会在事务最终失败前就记录成功 Event 的 Key。
                        observed_keys.add(key)
                        events.append(Event(key=key, value=value, is_write=is_write))
                committed_transaction = reader.boolean()
                if committed_transaction:
                    committed.append(
                        Transaction(
                            transaction_id=transaction_id,
                            session_id=session_id,
                            session_position=len(committed),
                            events=tuple(events),
                        )
                    )
            sessions.append(Session(session_id=session_id, transactions=tuple(committed)))

        trailing = stream.read(1)
        if trailing:
            raise HistoryFormatError(f"解析完成后仍有额外数据，起始偏移为 {reader.offset}")

    # 与 VeriStrong 一致，初始事务最后加入 History，但使用 ID 0 和 Session 0。
    initial_transaction = Transaction(
        transaction_id=0,
        session_id=0,
        session_position=0,
        events=tuple(Event(key=key, value=0, is_write=True) for key in sorted(observed_keys)),
        is_initial=True,
    )
    sessions.append(Session(session_id=0, transactions=(initial_transaction,)))
    return DbcopHistory(
        header=header,
        sessions=tuple(sessions),
        observed_keys=tuple(sorted(observed_keys)),
        source=str(source.resolve()),
    )


def compute_transaction_accesses(history: DbcopHistory) -> dict[int, TransactionAccess]:
    """提取每个事务的第一次外部读和最后一次写。"""

    result: dict[int, TransactionAccess] = {}
    for transaction in history.transactions:
        current_writes: dict[int, int] = {}
        reads: dict[int, int] = {}
        for event in transaction.events:
            if event.is_write:
                current_writes[event.key] = event.value
            elif event.key not in current_writes:
                reads.setdefault(event.key, event.value)
            elif current_writes[event.key] != event.value:
                raise ConstraintConstructionError(
                    f"事务 {transaction.transaction_id} 在 Key {event.key} 上违反 Int："
                    f"事务内写入 {current_writes[event.key]} 后读取 {event.value}"
                )
        result[transaction.transaction_id] = TransactionAccess(
            reads=dict(reads),
            writes=dict(current_writes),
        )
    return result


def _construct_choices(
    accesses: Mapping[int, TransactionAccess],
) -> tuple[tuple[WWChoice, ...], tuple[WRChoice, ...]]:
    writers_by_key: dict[int, set[int]] = defaultdict(set)
    writers_by_key_value: dict[tuple[int, int], set[int]] = defaultdict(set)
    for transaction_id, access in accesses.items():
        for key, value in access.writes.items():
            writers_by_key[key].add(transaction_id)
            writers_by_key_value[(key, value)].add(transaction_id)

    # VeriStrong 会把同一事务对共享的多个 Key 合并到一个 WWConstraint。
    keys_by_pair: dict[tuple[int, int], set[int]] = defaultdict(set)
    for key, writers in writers_by_key.items():
        ordered = sorted(writers)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                keys_by_pair[(left, right)].add(key)
    ww_choices = tuple(
        WWChoice(group_id=index, left_transaction=left, right_transaction=right, keys=tuple(sorted(keys)))
        for index, ((left, right), keys) in enumerate(sorted(keys_by_pair.items()))
    )

    raw_wr: list[tuple[int, int, int, tuple[int, ...]]] = []
    for read_transaction, access in accesses.items():
        for key, value in access.reads.items():
            candidates = tuple(
                writer
                for writer in sorted(writers_by_key_value[(key, value)])
                if writer != read_transaction
            )
            if not candidates:
                raise ConstraintConstructionError(
                    f"读取事务 {read_transaction} 在 Key {key} 上读取值 {value}，"
                    "但没有匹配的候选写事务"
                )
            raw_wr.append((read_transaction, key, value, candidates))
    wr_choices = tuple(
        WRChoice(
            group_id=index,
            key=key,
            value=value,
            read_transaction=read_transaction,
            candidate_writers=candidates,
        )
        for index, (read_transaction, key, value, candidates) in enumerate(sorted(raw_wr))
    )
    return ww_choices, wr_choices


def _is_acyclic(nodes: Sequence[int], edges: DependencyEdges) -> bool:
    adjacency = edges.adjacency(nodes)
    indegree = {node: 0 for node in adjacency}
    for targets in adjacency.values():
        for target in targets:
            indegree[target] += 1
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(indegree)


def _reachability(nodes: Sequence[int], edges: DependencyEdges) -> dict[int, set[int]]:
    """利用已知图为 DAG 的条件，以逆拓扑 bitset 计算传递闭包。"""

    adjacency = edges.adjacency(nodes)
    indegree = {node: 0 for node in nodes}
    for targets in adjacency.values():
        for target in targets:
            indegree[target] += 1
    queue = deque(sorted(node for node in nodes if indegree[node] == 0))
    topological: list[int] = []
    while queue:
        node = queue.popleft()
        topological.append(node)
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(topological) != len(nodes):
        raise ConstraintConstructionError("已知依赖图包含环，无法计算可达闭包")

    node_list = list(nodes)
    node_index = {node: index for index, node in enumerate(node_list)}
    closure = [1 << index for index in range(len(node_list))]
    for node in reversed(topological):
        index = node_index[node]
        for target in adjacency[node]:
            closure[index] |= closure[node_index[target]]

    result: dict[int, set[int]] = {}
    for node, encoded in zip(node_list, closure):
        reached: set[int] = set()
        while encoded:
            least_bit = encoded & -encoded
            reached.add(node_list[least_bit.bit_length() - 1])
            encoded ^= least_bit
        result[node] = reached
    return result


def _add_derived_rw_for_ww(
    edges: DependencyEdges,
    source: int,
    target: int,
    keys: Iterable[int],
    wr_by_writer_key: Mapping[tuple[int, int], set[int]],
) -> None:
    """加入 WW 后，从所有已固定 WR 推导对应 RW。"""

    for key in keys:
        for reader in wr_by_writer_key.get((source, key), ()):
            if reader != target:
                edges.add("RW", reader, target, (key,))


def _add_derived_rw_for_wr(
    edges: DependencyEdges,
    writer: int,
    reader: int,
    key: int,
    ww_by_writer_key: Mapping[tuple[int, int], set[int]],
) -> None:
    """加入 WR 后，从所有已固定 WW 推导对应 RW。"""

    for target in ww_by_writer_key.get((writer, key), ()):
        if target != reader:
            edges.add("RW", reader, target, (key,))


def _ww_orientation_valid(
    choice: WWChoice,
    left_to_right: bool,
    reachability: Mapping[int, set[int]],
    wr_by_writer_key: Mapping[tuple[int, int], set[int]],
) -> bool:
    source = choice.left_transaction if left_to_right else choice.right_transaction
    target = choice.right_transaction if left_to_right else choice.left_transaction
    if source in reachability[target]:
        return False
    for key in choice.keys:
        for reader in wr_by_writer_key.get((source, key), ()):
            if reader == target:
                continue
            if reader in reachability[target]:
                return False
    return True


def _wr_candidate_valid(
    choice: WRChoice,
    writer: int,
    reachability: Mapping[int, set[int]],
    ww_by_writer_key: Mapping[tuple[int, int], set[int]],
) -> bool:
    reader = choice.read_transaction
    if writer in reachability[reader]:
        return False
    for target in ww_by_writer_key.get((writer, choice.key), ()):
        if target != reader and reader in reachability[target]:
            return False
    return True


def construct_veristrong_problem(
    history: DbcopHistory,
    *,
    fast_prune: bool = True,
) -> VeriStrongProblem:
    """构造 VeriStrong WW/WR 选择，并可选复现 ``fast_prune_constraints``。"""

    accesses = compute_transaction_accesses(history)
    all_ww, all_wr = _construct_choices(accesses)
    nodes = tuple(sorted(accesses))
    fixed_edges = DependencyEdges()

    # Session 内只连接相邻提交事务，与 C++ known_graph_of() 一致。
    for session in history.sessions:
        for left, right in zip(session.transactions, session.transactions[1:]):
            fixed_edges.add("SO", left.transaction_id, right.transaction_id)

    remaining_wr: dict[int, WRChoice] = {choice.group_id: choice for choice in all_wr}
    wr_by_writer_key: dict[tuple[int, int], set[int]] = defaultdict(set)
    ww_by_writer_key: dict[tuple[int, int], set[int]] = defaultdict(set)
    # unit WR 在 fast pruning 第 0 步直接转成已知边。
    for choice in all_wr:
        if len(choice.candidate_writers) == 1:
            writer = choice.candidate_writers[0]
            fixed_edges.add("WR", writer, choice.read_transaction, (choice.key,))
            wr_by_writer_key[(writer, choice.key)].add(choice.read_transaction)
            remaining_wr.pop(choice.group_id)

    if not fast_prune:
        return VeriStrongProblem(
            history=history,
            accesses=accesses,
            all_ww_choices=all_ww,
            all_wr_choices=all_wr,
            fixed_edges=fixed_edges,
            remaining_ww_choices=all_ww,
            remaining_wr_choices=tuple(remaining_wr.values()),
        )

    remaining_ww: dict[int, WWChoice] = {choice.group_id: choice for choice in all_ww}
    rounds = 0
    changed = True
    while changed:
        rounds += 1
        changed = False
        if not _is_acyclic(nodes, fixed_edges):
            return VeriStrongProblem(
                history, accesses, all_ww, all_wr, fixed_edges,
                tuple(remaining_ww.values()), tuple(remaining_wr.values()), False, rounds,
            )
        reach = _reachability(nodes, fixed_edges)

        for group_id, choice in list(remaining_ww.items()):
            left_valid = _ww_orientation_valid(choice, True, reach, wr_by_writer_key)
            right_valid = _ww_orientation_valid(choice, False, reach, wr_by_writer_key)
            if not left_valid and not right_valid:
                return VeriStrongProblem(
                    history, accesses, all_ww, all_wr, fixed_edges,
                    tuple(remaining_ww.values()), tuple(remaining_wr.values()), False, rounds,
                )
            if left_valid == right_valid:
                continue
            source = choice.left_transaction if left_valid else choice.right_transaction
            target = choice.right_transaction if left_valid else choice.left_transaction
            fixed_edges.add("WW", source, target, choice.keys)
            _add_derived_rw_for_ww(
                fixed_edges, source, target, choice.keys, wr_by_writer_key
            )
            for key in choice.keys:
                ww_by_writer_key[(source, key)].add(target)
            remaining_ww.pop(group_id)
            changed = True

        for group_id, choice in list(remaining_wr.items()):
            candidates = tuple(
                writer
                for writer in choice.candidate_writers
                if _wr_candidate_valid(choice, writer, reach, ww_by_writer_key)
            )
            if not candidates:
                return VeriStrongProblem(
                    history, accesses, all_ww, all_wr, fixed_edges,
                    tuple(remaining_ww.values()), tuple(remaining_wr.values()), False, rounds,
                )
            if candidates != choice.candidate_writers:
                choice = WRChoice(
                    group_id=choice.group_id,
                    key=choice.key,
                    value=choice.value,
                    read_transaction=choice.read_transaction,
                    candidate_writers=candidates,
                )
                remaining_wr[group_id] = choice
            if len(candidates) == 1:
                writer = candidates[0]
                fixed_edges.add("WR", writer, choice.read_transaction, (choice.key,))
                _add_derived_rw_for_wr(
                    fixed_edges,
                    writer,
                    choice.read_transaction,
                    choice.key,
                    ww_by_writer_key,
                )
                wr_by_writer_key[(writer, choice.key)].add(choice.read_transaction)
                remaining_wr.pop(group_id)
                changed = True

    return VeriStrongProblem(
        history=history,
        accesses=accesses,
        all_ww_choices=all_ww,
        all_wr_choices=all_wr,
        fixed_edges=fixed_edges,
        remaining_ww_choices=tuple(remaining_ww.values()),
        remaining_wr_choices=tuple(remaining_wr.values()),
        pruning_consistent=True,
        pruning_rounds=rounds,
    )


def problem_summary(problem: VeriStrongProblem) -> dict[str, int | bool | str | None]:
    """生成适合日志和黄金样例比对的稳定统计。"""

    history = problem.history
    return {
        "source": history.source,
        "sessions": len(history.sessions),
        "transactions": len(history.transactions),
        "keys": len(history.observed_keys),
        "events": sum(len(tx.events) for tx in history.transactions),
        "so_fixed": len(problem.fixed_edges.by_type["SO"]),
        "wr_fixed": len(problem.fixed_edges.by_type["WR"]),
        "ww_fixed": len(problem.fixed_edges.by_type["WW"]),
        "rw_fixed": len(problem.fixed_edges.by_type["RW"]),
        "ww_groups_total": len(problem.all_ww_choices),
        "wr_groups_total": len(problem.all_wr_choices),
        "ww_groups_remaining": len(problem.remaining_ww_choices),
        "wr_groups_remaining": len(problem.remaining_wr_choices),
        "wr_candidates_remaining": sum(
            len(choice.candidate_writers) for choice in problem.remaining_wr_choices
        ),
        "pruning_consistent": problem.pruning_consistent,
        "pruning_rounds": problem.pruning_rounds,
    }


def export_problem_json(problem: VeriStrongProblem, path: str | Path) -> None:
    """导出稳定、可审阅的调试 JSON；批量训练应使用 Tensor 缓存。"""

    payload = {
        "format": "isolation-gnn-veristrong-v1",
        "summary": problem_summary(problem),
        "transactions": [
            {
                "id": tx.transaction_id,
                "session_id": tx.session_id,
                "session_position": tx.session_position,
                "is_initial": tx.is_initial,
                "events": [
                    {"key": event.key, "value": event.value, "type": "W" if event.is_write else "R"}
                    for event in tx.events
                ],
            }
            for tx in problem.history.transactions
        ],
        "fixed_edges": [
            {"type": kind, "source": source, "target": target, "keys": sorted(keys)}
            for kind, source, target, keys in problem.fixed_edges.typed_edges()
        ],
        "remaining_ww": [
            {
                "group_id": choice.group_id,
                "left": choice.left_transaction,
                "right": choice.right_transaction,
                "keys": list(choice.keys),
            }
            for choice in problem.remaining_ww_choices
        ],
        "remaining_wr": [
            {
                "group_id": choice.group_id,
                "key": choice.key,
                "value": choice.value,
                "reader": choice.read_transaction,
                "candidate_writers": list(choice.candidate_writers),
            }
            for choice in problem.remaining_wr_choices
        ],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "ConstraintConstructionError",
    "DbcopHeader",
    "DbcopHistory",
    "DependencyEdges",
    "Event",
    "HistoryFormatError",
    "Session",
    "Transaction",
    "TransactionAccess",
    "VeriStrongProblem",
    "WRChoice",
    "WWChoice",
    "compute_transaction_accesses",
    "construct_veristrong_problem",
    "export_problem_json",
    "parse_dbcop_history",
    "problem_summary",
]
