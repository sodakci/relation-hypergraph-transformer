"""第一版隔离级别决策图的关系模式。"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Literal

NodeType = Literal["transaction", "key", "decision"]

# 每个关系都明确规定源节点类型和目标节点类型。
# 反向关系不是简单的命名别名，而是拥有独立可学习参数的关系。
RELATION_ENDPOINTS: Final = MappingProxyType(
    {
        # Transaction -> Transaction：已经确定的事务依赖。
        "SO": ("transaction", "transaction"),
        "SO_REV": ("transaction", "transaction"),
        "WR": ("transaction", "transaction"),
        "WR_REV": ("transaction", "transaction"),
        "WW_FIXED": ("transaction", "transaction"),
        "WW_FIXED_REV": ("transaction", "transaction"),
        "RW": ("transaction", "transaction"),
        "RW_REV": ("transaction", "transaction"),
        # Transaction <-> Key：描述读写访问，使共享 Key 的事务可以交换信息。
        "READS": ("transaction", "key"),
        "READ_BY": ("key", "transaction"),
        "WRITES": ("transaction", "key"),
        "WRITTEN_BY": ("key", "transaction"),
        # Transaction <-> Decision：区分决策变量的左、右事务。
        "LEFT_OF": ("transaction", "decision"),
        "RIGHT_OF": ("transaction", "decision"),
        "LEFT_TRANSACTION": ("decision", "transaction"),
        "RIGHT_TRANSACTION": ("decision", "transaction"),
    }
)

# 元组顺序同时决定模型创建关系参数时的稳定顺序。
DEFAULT_RELATIONS: Final[tuple[str, ...]] = tuple(RELATION_ENDPOINTS)
