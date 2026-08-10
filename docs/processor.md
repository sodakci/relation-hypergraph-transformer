# Recurrent Processor 输入编码与深度消融实验

本文首先说明 `history.bincode` 和 `labels.json` 如何转换为
`VeriStrongDecisionNetwork` 的张量输入，然后记录不同 Processor Steps 的消融结果。
当前网络只预测 fast pruning 后剩余的 WW/WR 选择，但用于预测的图上下文同时包含固定
SO、WR、WW、RW、Key 访问关系、候选角色和选择组关系。

## 1. 完整输入链条

训练时存在两条输入支路：历史负责构造图，VeriStrong 标签负责构造监督目标。

```text
history.bincode
    ├─ 解析 Session / Transaction / Event
    ├─ 过滤失败 Event 和未提交 Transaction，并加入初始事务 T0
    ├─ 提取第一次外部读和最后一次写
    ├─ 构造 SO、WW、WR 约束
    ├─ fast pruning，固定部分 WW/WR，并同步推导 RW
    └─ 构造四类节点、24 类关系边和 Decision 辅助索引
                         │
                         ▼
                    VeriStrongGraph
                         │
labels.json              │
    ├─ 校验 producer     │
    ├─ 校验 fingerprint  │
    └─ 原始 group/事务 ID ──→ VeriStrongTargets
                         │
                         ▼
                 DataLoader 批处理拼接
                         │
                         ▼
              图和标签移动到训练设备
                         │
                         ▼
             使用训练集统计量标准化节点特征
                         │
                         ▼
       四类 NodeEncoder → Recurrent Processor → WW/WR Head
                         │
                         ▼
               WW BCE + WR grouped loss
```

`history.bincode` 不会以原始字节形式直接输入网络。原始事务 ID、Key ID 和 Value
也不会作为连续数值特征直接送入 Linear：ID 用于稳定映射，Value 用于匹配 WR 候选，
真正输入模型的是图结构、节点统计特征和紧凑索引。

对应实现：

- 历史和约束：[`veristrong_data.py`](../src/isolation_gnn/veristrong_data.py)
- 构图和索引：[`veristrong_graph.py`](../src/isolation_gnn/veristrong_graph.py)
- 标签到 Target：[`veristrong_loss.py`](../src/isolation_gnn/veristrong_loss.py)
- 标准化：[`veristrong_normalize.py`](../src/isolation_gnn/veristrong_normalize.py)
- 网络前向：[`veristrong_model.py`](../src/isolation_gnn/veristrong_model.py)

## 2. `VeriStrongGraph` 的完整字段

设：

- `Nt`：Transaction 节点数；
- `Nk`：Key 节点数；
- `Nd`：Decision 节点数；
- `Nc`：Constraint 节点数；
- `Nww`：fast pruning 后剩余 WW group 数；
- `Nwr`：fast pruning 后剩余 WR group 数；
- `Nwrc`：所有剩余 WR group 的 candidate writer 总数。

其中：

```text
Nd = Nww + Nwrc
Nc = Nww + Nwr
```

完整图输入如下：

| 字段 | 类型/形状 | 含义 |
|---|---|---|
| `transaction_features` | `FloatTensor[Nt,16]` | Transaction 节点特征 |
| `key_features` | `FloatTensor[Nk,6]` | Key 节点特征 |
| `decision_features` | `FloatTensor[Nd,16]` | WW group 或 WR candidate 特征 |
| `constraint_features` | `FloatTensor[Nc,6]` | WW/WR 选择组特征 |
| `relation_edges` | `dict[str, (LongTensor[Er], LongTensor[Er])]` | 每类关系的 source/target 稀疏边索引 |
| `decision_left_transaction` | `LongTensor[Nd]` | Decision 的 left 事务局部索引 |
| `decision_right_transaction` | `LongTensor[Nd]` | Decision 的 right 事务局部索引 |
| `decision_constraint` | `LongTensor[Nd]` | Decision 所属 Constraint 的局部索引 |
| `decision_wr_group` | `LongTensor[Nd]` | WR 紧凑分组编号；WW 为 `-1` |
| `ww_mask` | `BoolTensor[Nd]` | 哪些 Decision 属于 WW |
| `wr_mask` | `BoolTensor[Nd]` | 哪些 Decision 属于 WR |
| `transaction_ids` | `tuple[int,...]`，长度 `Nt` | Transaction 局部索引到原始事务 ID 的反向映射 |
| `key_ids` | `tuple[int,...]`，长度 `Nk` | Key 局部索引到原始 Key ID 的反向映射 |
| `ww_choice_ids` | `tuple[int,...]`，长度 `Nww` | 紧凑 WW 输出顺序到原始 WW group ID 的映射 |
| `wr_choice_ids` | `tuple[int,...]`，长度 `Nwr` | WR 紧凑组编号到原始 WR group ID 的映射 |

`relation_edges` 参与消息传播；六个 `decision_*`/mask 张量用于从传播后的状态中组装
WW/WR 输出头上下文；四个 `*_ids` 元组只用于标签对齐、输出还原和调试，不参与可学习计算。

## 3. 三套编号必须区分

### 3.1 原始 ID

原始 ID 来自历史和剪枝前的约束，例如：

```text
Transaction ID：0、2、5、9、12
Key ID：7、20
WW group ID：4、7
WR group ID：3、8
```

未提交事务会被过滤，但解析器仍为它消耗原始事务编号，因此已提交事务 ID 可能不连续。
fast pruning 还会移除已固定的 group，所以剩余 WW/WR group ID 也可能不连续。

### 3.2 类型内局部索引

为了直接索引紧凑张量，每类节点都使用从 0 开始的连续编号：

```text
Transaction：0 ... Nt-1
Key：        0 ... Nk-1
Decision：   0 ... Nd-1
Constraint： 0 ... Nc-1
```

例如：

```python
transaction_ids = (0, 2, 5, 9, 12)
```

表示：

```text
transaction_features[0] 对应原始 T0
transaction_features[1] 对应原始 T2
transaction_features[2] 对应原始 T5
transaction_features[3] 对应原始 T9
transaction_features[4] 对应原始 T12
```

构图时建立正向映射：

```python
transaction_index = {
    raw_transaction_id: local_index
    for local_index, raw_transaction_id in enumerate(transaction_ids)
}
```

所以原始事务序列 `[2,5,0,5,2,12]` 编码后是局部行号
`[1,2,0,2,1,4]`。`decision_left_transaction` 和
`decision_right_transaction` 保存的是后者，因为模型需要直接执行：

```python
left_state = transaction_state[decision_left_transaction]
```

恢复原始 ID：

```python
raw_id = transaction_ids[local_index]
```

### 3.3 全局消息传播索引

四类节点编码到相同隐藏维度 `H` 后按以下顺序拼接：

```text
states = [Transaction | Key | Decision | Constraint]
```

全局偏移为：

```text
Transaction offset = 0
Key offset         = Nt
Decision offset    = Nt + Nk
Constraint offset  = Nt + Nk + Nd
```

`relation_edges` 中保存的是这个全局编号；Decision 的辅助索引仍是类型内局部编号，
因为输出头先从 `states` 切出 `transaction_state`、`decision_state` 和
`constraint_state`，再执行类型内索引。

## 4. 四类节点特征

### 4.1 Transaction 编码 `[Nt,16]`

| 位置 | 特征 |
|---:|---|
| 0 | `log1p(读事件数量)` |
| 1 | `log1p(写事件数量)` |
| 2 | `log1p(总事件数量)` |
| 3 | `log1p(访问的不同 Key 数量)` |
| 4 | 事务在 Session 中的相对位置 |
| 5 | `log1p(Session 长度)` |
| 6 | 固定 SO 入度 |
| 7 | 固定 SO 出度 |
| 8 | 固定 WR 入度 |
| 9 | 固定 WR 出度 |
| 10 | 固定 WW 入度 |
| 11 | 固定 WW 出度 |
| 12 | 固定 RW 入度 |
| 13 | 固定 RW 出度 |
| 14 | 能到达当前事务的事务比例 |
| 15 | 当前事务能够到达的事务比例 |

### 4.2 Key 编码 `[Nk,6]`

| 位置 | 特征 |
|---:|---|
| 0 | `log1p(不同 reader 事务数)` |
| 1 | `log1p(不同 writer 事务数)` |
| 2 | `log1p(读事件总数)` |
| 3 | `log1p(写事件总数)` |
| 4 | `log1p(读写事件总数)` |
| 5 | `log1p(访问该 Key 的不同事务数)` |

Key 统计不包含初始化所有 Key 的初始事务 T0，避免初始化写入淹没真实工作负载统计。

### 4.3 Decision 编码 `[Nd,16]`

| 位置 | 特征 |
|---:|---|
| 0 | 是否为 WW |
| 1 | 是否为 WR |
| 2 | left/right 是否在同一个 Session |
| 3 | left/right 是否位于可能依赖图的同一强连通分量 |
| 4 | `log1p(group 候选数量)` |
| 5 | `log1p(涉及 Key 数量)` |
| 6 | `log1p(left 的固定依赖度数)` |
| 7 | `log1p(right 的固定依赖度数)` |
| 8 | 固定图中 left 是否已经可达 right |
| 9 | 固定图中 right 是否已经可达 left |
| 10 | left/right 在 Session 中的位置差 |
| 11 | `log1p(相同值的候选 writer 数量)` |
| 12 | left 是否写入相关 Key |
| 13 | right 是否读取相关 Key |
| 14 | `log1p(left/right 参与剩余候选的总次数)` |
| 15 | left/right 是否包含初始事务 T0 |

角色规则：

| Decision 类型 | left | right | 一个 group 创建几个 Decision |
|---|---|---|---:|
| WW | 左侧 writer | 右侧 writer | 1 |
| WR | candidate writer | reader | 每个 candidate writer 创建 1 个 |

### 4.4 Constraint 编码 `[Nc,6]`

| 位置 | 特征 |
|---:|---|
| 0 | 是否为 WW 约束 |
| 1 | 是否为 WR 约束 |
| 2 | `log1p(候选数量)` |
| 3 | 候选中是否包含初始事务 T0 |
| 4 | 根据固定可达关系计算的不可行候选比例 |
| 5 | 预留比例特征，当前为 0 |

每个剩余 WW group 创建一个 Constraint；每个剩余 WR group 也创建一个 Constraint，
同组所有 WR candidate Decision 共享该 Constraint。

## 5. `relation_edges`：24 类稀疏关系

### 5.1 为什么它是字典而不是一个矩阵

`relation_edges` 的类型是：

```python
dict[str, tuple[LongTensor, LongTensor]]
```

例如：

```python
relation_edges["SO"] = (
    tensor([0, 1, 3]),  # source，形状 [E_SO]
    tensor([1, 2, 4]),  # target，形状 [E_SO]
)
```

它表示 `0→1、1→2、3→4` 三条 SO 边。两个一维向量等价于常见的
`edge_index[2,E]` COO 稀疏表示，不创建会占用 `O(N²)` 空间的稠密邻接矩阵。

字典的 key 保留关系语义，使 Processor 能为每种关系使用不同参数：

```text
SO         使用 W_SO
WR_FIXED   使用 W_WR_FIXED
READS      使用 W_READS
WR_WRITER  使用 W_WR_WRITER
...
```

一个具体历史没有某类边时，该关系可能不出现在当前字典中；网络配置的关系全集仍为24类。

### 5.2 24 类关系

| 编号 | 关系 | 方向 | 含义 |
|---:|---|---|---|
| 1 | `SO` | Transaction → Transaction | 固定 Session Order |
| 2 | `SO_REV` | Transaction → Transaction | SO 反向消息边 |
| 3 | `WR_FIXED` | Transaction → Transaction | 已固定 writer → reader |
| 4 | `WR_FIXED_REV` | Transaction → Transaction | 固定 WR 反向消息边 |
| 5 | `WW_FIXED` | Transaction → Transaction | 已固定前写事务 → 后写事务 |
| 6 | `WW_FIXED_REV` | Transaction → Transaction | 固定 WW 反向消息边 |
| 7 | `RW_FIXED` | Transaction → Transaction | 已推导 reader → 后续 writer |
| 8 | `RW_FIXED_REV` | Transaction → Transaction | 固定 RW 反向消息边 |
| 9 | `READS` | Transaction → Key | 事务读取该 Key |
| 10 | `READ_BY` | Key → Transaction | `READS` 的反向消息边 |
| 11 | `WRITES` | Transaction → Key | 事务写入该 Key |
| 12 | `WRITTEN_BY` | Key → Transaction | `WRITES` 的反向消息边 |
| 13 | `WW_LEFT` | Transaction → Decision | WW 左侧事务连接到决策 |
| 14 | `WW_LEFT_REV` | Decision → Transaction | WW 左侧反向消息边 |
| 15 | `WW_RIGHT` | Transaction → Decision | WW 右侧事务连接到决策 |
| 16 | `WW_RIGHT_REV` | Decision → Transaction | WW 右侧反向消息边 |
| 17 | `WR_WRITER` | Transaction → Decision | candidate writer 连接到 WR 决策 |
| 18 | `WR_WRITER_REV` | Decision → Transaction | WR writer 反向消息边 |
| 19 | `WR_READER` | Transaction → Decision | reader 连接到 WR 决策 |
| 20 | `WR_READER_REV` | Decision → Transaction | WR reader 反向消息边 |
| 21 | `DECISION_KEY` | Decision → Key | 决策涉及该 Key |
| 22 | `KEY_DECISION` | Key → Decision | Key 到决策的反向消息边 |
| 23 | `MEMBER_OF` | Decision → Constraint | 决策属于某个选择组 |
| 24 | `HAS_MEMBER` | Constraint → Decision | 选择组包含该决策 |

24类关系提供预测 WW/WR 所需的完整上下文。模型的输出目标只有 WW/WR，但如果没有
SO、固定 WR/WW/RW 和 Key 访问关系，模型就看不到已有可达性与潜在成环风险；如果没有
Decision/Constraint 关系，同一 WR group 的 candidate writer 也无法交换分组信息。

## 6. Decision 辅助索引的完整编码规则

### 6.1 构造顺序

当前实现先遍历 `remaining_ww_choices`，再遍历 `remaining_wr_choices`：

```text
Decision 0 ... Nww-1          ：每个剩余 WW group 一个 Decision
Decision Nww ... Nd-1         ：每个剩余 WR candidate writer 一个 Decision

Constraint 0 ... Nww-1        ：每个剩余 WW group 一个 Constraint
Constraint Nww ... Nc-1       ：每个剩余 WR group 一个 Constraint
```

### 6.2 六个张量

| 字段 | WW Decision | WR Decision | 主要用途 |
|---|---|---|---|
| `decision_left_transaction[d]` | `transaction_index[left_writer]` | `transaction_index[candidate_writer]` | 读取 left/candidate writer 状态 |
| `decision_right_transaction[d]` | `transaction_index[right_writer]` | `transaction_index[reader]` | 读取 right/reader 状态 |
| `decision_constraint[d]` | 当前 WW Constraint 局部编号 | 当前 WR Constraint 局部编号 | 读取选择组状态 |
| `decision_wr_group[d]` | `-1` | `0...Nwr-1` 的紧凑 WR 组编号 | WR 分组 loss 和输出还原 |
| `ww_mask[d]` | `True` | `False` | 选择进入 WW Head 的 Decision |
| `wr_mask[d]` | `False` | `True` | 选择进入 WR Head 的 Decision |

其中：

```python
wr_mask = ~ww_mask
```

每个 Decision 必须且只能属于一种类型。`decision_left_transaction` 等名称中的
`transaction` 表示它索引 Transaction 状态矩阵，并不表示张量中直接保存原始事务 ID。

### 6.3 四个反向映射字段

| 字段 | 编码规则 | 恢复规则 |
|---|---|---|
| `transaction_ids[i]` | 第 `i` 行 Transaction 对应的原始事务 ID | `raw_tx = transaction_ids[local_tx]` |
| `key_ids[i]` | 第 `i` 行 Key 对应的原始 Key ID | `raw_key = key_ids[local_key]` |
| `ww_choice_ids[k]` | 第 `k` 个 WW logit 对应的原始 WW group ID | `raw_ww = ww_choice_ids[k]` |
| `wr_choice_ids[g]` | 紧凑 WR group `g` 对应的原始 WR group ID | `raw_wr = wr_choice_ids[decision_wr_group[d]]` |

WW 的严格对应方法为：

```python
ww_indices = torch.nonzero(ww_mask, as_tuple=False).squeeze(-1)
for compact_ww, decision_index in enumerate(ww_indices):
    raw_group_id = ww_choice_ids[compact_ww]
```

WR 的 `wr_choice_ids` 是每个 group 一个，不是每个 candidate Decision 一个。同组多个
Decision 通过相同的 `decision_wr_group` 映射到同一个原始 group ID。

### 6.4 为什么同一个 group 存在多套编号

这里不是重复保存同一个值，而是分别保存四个索引空间：

| 编号 | 示例 | 索引对象 | 用途 |
|---|---:|---|---|
| Decision 局部编号 | `D2` | `decision_state[2]` | 选择一个具体候选节点的隐藏状态 |
| Constraint 局部编号 | `C2` | `constraint_state[2]` | 选择该候选所属组的隐藏状态 |
| WR 紧凑组编号 | `0` | 分组 loss 的第0组 | 把同组多个 WR candidate 聚合到一起 |
| 原始 WR group ID | `3` | VeriStrong 约束、`labels.json` | 与求解器标签和 witness 对齐 |

例如：

```python
wr_choice_ids = (3, 8)
```

表示紧凑 WR group 与原始 ID 的映射为：

```text
dense WR group 0 → 原始 WR group 3
dense WR group 1 → 原始 WR group 8
```

假设前两个 Decision 是 WW，后四个 Decision 是两个 WR group 的候选：

```python
decision_wr_group = tensor([-1, -1, 0, 0, 1, 1])
```

那么：

```text
D0、D1：WW，不属于 WR group，所以为 -1
D2、D3：属于 dense WR group 0，对应原始 group 3
D4、D5：属于 dense WR group 1，对应原始 group 8
```

从一个 WR Decision 恢复原始 group ID 的完整公式是：

```python
dense_group = int(decision_wr_group[decision_index])
raw_group_id = wr_choice_ids[dense_group]
```

WR 必须显式保存 `decision_wr_group`，因为一个 group 会产生多个 candidate Decision；
分组交叉熵需要知道哪些 logits 应放在同一个 softmax 分母中。这里使用连续的
`0...Nwr-1`，而不直接存原始 group ID，原因是：

- fast pruning 后原始 group ID 可能有空洞；
- 不同 history 的原始 group ID 可能重复；
- `scatter_add`/`scatter_reduce` 等分组张量操作要求紧凑连续索引更高效；
- batch 时只需给后续图增加 WR group 数量偏移。

WW 没有单独的 `decision_ww_group`，因为当前结构中一个剩余 WW group 恰好对应一个
Decision。它使用 `ww_mask` 取出 WW Decision 后，依靠顺序隐式对齐：

```python
ww_indices = torch.nonzero(ww_mask, as_tuple=False).squeeze(-1)

# 第 k 个 WW logit：
decision_index = ww_indices[k]
raw_ww_group_id = ww_choice_ids[k]
```

因此：

```python
ww_choice_ids = (4, 7)
```

表示第0个 WW logit 对应原始 group 4，第1个 WW logit 对应原始 group 7；数值4和7
不会拿来索引 `decision_state`。如果以后把一个 WW group 展开成多个候选 Decision，
则也应像 WR 一样增加显式的 `decision_ww_group` 紧凑索引。

`decision_constraint` 又是另一套编号。它索引的是统一的 Constraint 状态矩阵，其中先放
WW Constraint，再放 WR Constraint。因此在单图当前构造顺序下：

```text
WW group 的 Constraint index = 0 ... Nww-1
WR dense group g 的 Constraint index = Nww + g
```

它的作用是读取组节点表示，不用于恢复 VeriStrong 原始 group ID。

### 6.5 完整示例

假设原始对象为：

```text
Transaction IDs = {0,2,5,9,12}
Key IDs = {7,20}

剩余 WW：
    group 4：T2 与 T9
    group 7：T5 与 T12

剩余 WR：
    group 3：reader=T12，candidate writers={T0,T5}
    group 8：reader=T9， candidate writers={T2,T12}
```

Transaction 局部映射为：

```text
T0→0，T2→1，T5→2，T9→3，T12→4
```

Decision 顺序为：

```text
D0 = WW group 4：left=T2， right=T9
D1 = WW group 7：left=T5， right=T12
D2 = WR group 3：writer=T0， reader=T12
D3 = WR group 3：writer=T5， reader=T12
D4 = WR group 8：writer=T2， reader=T9
D5 = WR group 8：writer=T12，reader=T9
```

最终编码：

```python
decision_left_transaction  = tensor([1, 2, 0, 2, 1, 4], dtype=torch.long)
decision_right_transaction = tensor([3, 4, 4, 4, 3, 3], dtype=torch.long)
decision_constraint        = tensor([0, 1, 2, 2, 3, 3], dtype=torch.long)
decision_wr_group          = tensor([-1, -1, 0, 0, 1, 1], dtype=torch.long)
ww_mask                    = tensor([True, True, False, False, False, False])
wr_mask                    = tensor([False, False, True, True, True, True])

transaction_ids = (0, 2, 5, 9, 12)
key_ids         = (7, 20)
ww_choice_ids   = (4, 7)
wr_choice_ids   = (3, 8)
```

例如 `decision_left_transaction[5] == 4`，恢复原始事务：

```python
transaction_ids[4] == 12
```

所以 D5 的 candidate writer 是原始 T12。若把原始 ID 12 直接写入
`decision_left_transaction`，模型会尝试读取 `transaction_state[12]`；当前只有5个
Transaction 节点，这会发生越界。

### 6.6 批处理时的偏移

多个历史组成 batch 时，它们之间不会增加任何关系边，只进行类型内拼接和索引平移：

- `decision_left_transaction`、`decision_right_transaction` 增加前面图的 Transaction 数；
- `decision_constraint` 增加前面图的 Constraint 数；
- WR 的 `decision_wr_group` 增加前面图的 WR group 数；
- WW 的 `decision_wr_group == -1` 保持不变；
- `ww_mask`、`wr_mask` 直接拼接；
- 四个 `*_ids` 元组按图顺序拼接。

## 7. 标签 Target 与标准化

### 7.1 标签 Target

`labels.json` 使用原始 group/事务 ID；加载后通过上面的反向映射对齐为：

| Target | 类型/形状 | 含义 |
|---|---|---|
| `ww_labels` | `FloatTensor[Nww]` | WW 方向，`1=True(left→right)`、`0=False(right→left)` |
| `ww_labeled_mask` | `BoolTensor[Nww]` | 哪些 WW 有监督标签 |
| `wr_feasible_mask` | `BoolTensor[Nwrc]` | 每个 WR candidate writer 是否属于标签可行集合 |
| `wr_labeled_group_mask` | `BoolTensor[Nwr]` | 哪些 WR group 有监督标签 |

当前 VeriStrong SAT model 是单解标签，因此每个有标签的 WR group 通常只有一个
`wr_feasible_mask=True`。图中的原始 ID 元组不进入网络，只保证标签和候选的语义对齐。

### 7.2 标准化

原始计数在构图阶段先做 `log1p`。之后仅使用训练集拟合 mean/std，并执行：

```text
x_normalized = (x - mean) / std
```

布尔、one-hot、相对位置、可达比例和方向位置差保持原值。标准化器与模型一起写入
checkpoint；验证和推理必须加载训练时统计，不能在测试集重新拟合。

### 7.3 进入 Processor 和输出头

四类 NodeEncoder 先映射到相同隐藏维度：

```text
Transaction：16 → H
Key：          6 → H
Decision：    16 → H
Constraint：   6 → H
```

随后拼接为 `[Nt+Nk+Nd+Nc,H]`，Processor 对每种关系分别变换和聚合：

```text
m_v = Σ_r mean_{u→v ∈ E_r}(W_r h_u)
```

传播结束后，Decision 辅助索引组装输出头输入：

```python
left_state  = transaction_state[decision_left_transaction]
right_state = transaction_state[decision_right_transaction]
group_state = constraint_state[decision_constraint]
```

WW/WR 两个 Head 分别由 `ww_mask`、`wr_mask` 选择 Decision，单个候选上下文为：

```text
[Decision, left, right, left-right, left×right, Constraint] = 6H
```

## 8. 深度消融实验参数

| 参数 | 设置 |
|---|---|
| 模型主干 | Recurrent Relational Processor |
| Hidden Dimension | 64 |
| Processor Steps | 3 / 6 / 9 / 12 |
| 训练 Epoch | 20 |
| 训练样本数 | 102 |
| 测试样本数 | 34 |
| 随机种子 | 7 |
| 推理设备 | CUDA:0（RTX 4090） |
| WW 测试样本 | 62,946 |
| WR 测试组数 | 34,940 |
| Beam Size | 8 |
| Max Expansions | 100,000 |
| 其他模型参数 | 保持一致，仅改变 Processor Steps |

## 9. WW / WR 预测结果

| Steps | Loss | WW Accuracy | WW Precision | WW Recall | WW F1 | WR Top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.8318 | 84.22% | 82.61% | **88.38%** | 85.40% | 78.21% |
| 6 | 0.8138 | 84.18% | 82.44% | **88.54%** | 85.38% | 78.24% |
| 9 | 0.8259 | 84.63% | 84.33% | 86.64% | **85.47%** | 78.37% |
| 12 | **0.8047** | **84.63%** | **85.81%** | 84.53% | 85.17% | **78.50%** |

增加传播轮数后，WW Accuracy 从 84.22% 小幅提高到约 84.63%，WR Top-1 从
78.21% 提升至 78.50%。其中：

- S9 的 WW F1 最高：85.47%；
- S12 的 WW Accuracy、Precision 和 WR Top-1 最高；
- S3/S6 更偏向高 Recall，S12 更偏向高 Precision；
- 整体提升存在，但幅度较小。

## 10. 完整 Witness / Beam Decode

| Steps | Exact Match | Direct Witness | Certified SAT | Certified Rate |
|---:|---:|---:|---:|---:|
| 3 | 0/34 | 0/34 | 3/34 | 8.82% |
| 6 | 0/34 | 0/34 | 4/34 | **11.76%** |
| 9 | 0/34 | 0/34 | 4/34 | **11.76%** |
| 12 | 0/34 | 0/34 | 4/34 | **11.76%** |

传播轮数从3增加到6后，Beam Certified SAT 从8.82%提高到11.76%。继续增加到9、
12轮后没有进一步提高，说明单纯增加传播深度已经开始出现收益饱和。

## 11. 推理性能

| Steps | Graph Construct | Model Inference | Beam Decode | 总流程约耗时 |
|---:|---:|---:|---:|---:|
| 3 | 10.67 s | **21.97 ms** | 10.27 s | 20.97 s |
| 6 | 10.69 s | 31.90 ms | **9.48 s** | **20.20 s** |
| 9 | **10.64 s** | 41.72 ms | 10.90 s | 21.59 s |
| 12 | 10.69 s | 51.05 ms | 11.03 s | 21.77 s |

模型推理成本随传播轮数近似线性增加：

```text
22 ms → 32 ms → 42 ms → 51 ms
```

但相对于图构建和 Beam Decode 的秒级耗时，神经网络本身仍不是主要性能瓶颈。

## 12. Hidden Dimension 消融实验参数

| 参数 | 设置 |
|---|---|
| 模型主干 | Recurrent Relational Processor |
| Hidden Dimension | 64 / 128 / 256 |
| Processor Steps | 9（固定） |
| 训练 Epoch | 20 |
| 训练样本数 | 102 |
| 测试样本数 | 34 |
| 随机种子 | 7 |
| 推理设备 | CUDA:0（RTX 4090） |
| WW 测试样本 | 62,946 |
| WR 测试组数 | 34,940 |
| Beam Size | 8 |
| Max Expansions | 100,000 |
| 其他模型参数 | 保持一致，仅改变 Hidden Dimension |

根据第 11 节结论将 Processor Steps 固定为 9，只改变模型宽度
Hidden Dimension，对应三个 checkpoint：`veristrong-h64-s9.pt`、
`veristrong-h128-s9.pt` 和 `veristrong-h256-s9.pt`。

## 13. Hidden Dimension WW / WR 预测结果

| Hidden | Loss | WW Accuracy | WW Precision | WW Recall | WW F1 | WR Top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.8259 | 84.63% | 84.33% | 86.64% | 85.47% | 78.37% |
| 128 | 0.7898 | 84.68% | 83.28% | **88.40%** | **85.76%** | 78.86% |
| 256 | **0.7811** | **84.93%** | **85.07%** | 86.26% | 85.66% | **79.01%** |

增加 Hidden Dimension 后 Loss 从 0.8259 持续下降到 0.7811，WW Accuracy 从
84.63% 提高到 84.93%，WR Top-1 从 78.37% 提升到 79.01%。其中：

- H128 的 WW Recall 和 WW F1 最高；
- H256 的 Loss、WW Accuracy、Precision 和 WR Top-1 最高；
- H64 → H128 的提升主要来自 WW Recall，H128 → H256 的提升主要来自 Precision；
- 整体提升仍然有限，继续加宽开始出现收益饱和。

## 14. Hidden Dimension 完整 Witness / Beam Decode

| Hidden | Exact Match | Direct Witness | Certified SAT | Certified Rate |
|---:|---:|---:|---:|---:|
| 64 | 0/34 | 0/34 | 4/34 | **11.76%** |
| 128 | 0/34 | 0/34 | 4/34 | **11.76%** |
| 256 | 0/34 | 1/34 | 3/34 | 8.82% |

H64 和 H128 的 Beam Certified SAT 均为 4/34。H256 首次出现 1 条直接 Witness
（`5_100_15_5000_0.5_r_0.5_0.5_100`），说明更大容量偶尔能让 Top-1 解码直接给出
合法 witness，但整体 Certified SAT 反而从 4/34 回落到 3/34，认证结果不稳定。

## 15. Hidden Dimension 推理性能

| Hidden | Graph Construct | Model Inference | Beam Decode | 总流程约耗时 |
|---:|---:|---:|---:|---:|
| 64 | **10.64 s** | **41.72 ms** | **10.90 s** | **21.59 s** |
| 128 | 10.71 s | 42.44 ms | 11.08 s | 21.84 s |
| 256 | 10.65 s | 51.98 ms | 11.03 s | 21.74 s |

模型推理成本随 Hidden Dimension 增加而上升：

```text
42 ms → 42 ms → 52 ms
```

但图构建和 Beam Decode 仍占秒级耗时，总流程时间基本不受 Hidden Dimension 影响，
神经网络本身仍不是主要性能瓶颈。

## 16. 实验结论

| 结论 | 判断 |
|---|---|
| 增加传播轮数是否有效 | 有一定效果 |
| S3 是否明显不足 | 是 |
| 继续单纯增加 Steps 是否值得 | 收益已经开始饱和 |
| 增加 Hidden Dimension 是否有效 | 有小幅提升，收益同样接近饱和 |
| WW 最佳综合结果 | **S9 + H128** |
| WR / Precision 最佳 | **S9 + H256** |
| 性能与效果折中 | **S9 + H128** |

两轮消融实验说明，增加传播轮数或 Hidden Dimension 都能小幅改善 WW/WR 预测，但
改善幅度有限。从 S6 开始完整 Certified SAT 已进入平台期（4/34）；Hidden Dimension
从 64 提高到 128 后 WW F1 从 85.47% 提升到 85.76%、WR Top-1 从 78.37% 提升到
78.86%，继续提高到 256 后 WW F1 基本持平、WR Top-1 微升到 79.01%，但 Certified
SAT 回落到 3/34，并出现 1 条直接 Witness。因此不建议继续尝试 H512 等更宽网络。

推荐将 Processor Steps 固定为 9、Hidden Dimension 固定为 128 作为下一阶段基线
（对应 `veristrong-h128-s9.pt`）。下一步应优先改进约束感知解码或标签质量，而不是
继续单纯扩大网络容量。
