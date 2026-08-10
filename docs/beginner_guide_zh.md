# VeriStrong Neural Solver 零基础入门手册

这份手册写给第一次接触 Python、PyTorch、图神经网络和本项目的读者。你暂时不需要理解数学公式，也不需要一次读完全部源码。先建立一张“地图”，以后看到陌生代码时，知道它处于哪一站、负责什么即可。

## 1. 这个项目到底在做什么

可以先把数据库中的很多事务想象成很多辆车：每辆车都要读取或修改一些数据。多辆车同时前进时，必须判断它们能否排成一个不会互相冲突的合法顺序。

这个项目的工作可以概括为：

> 读取一段数据库执行历史，预测尚未确定的事务依赖方向，再搜索并严格验证一个合法结果。

完整流程如下：

```text
history.bincode
      │
      ▼
解析历史、构造约束、确定性剪枝
      │
      ▼
构造成图（节点 + 边 + 特征）
      │
      ▼
GNN 给 WW / WR 候选打分
      │
      ▼
Beam Search 根据分数寻找完整选择
      │
      ▼
独立校验器检查结果是否合法
      │
      ├── CERTIFIED_SAT：找到并验证了合法结果
      ├── CERTIFIED_UNSAT：确定性剪枝已经证明无解
      └── UNKNOWN：当前预算内没找到，不能说一定无解
```

这里最重要的一点是：神经网络只负责“给建议、排优先级”，最终结果仍由确定性规则校验。模型分数高不等于结果自动正确。

## 2. 先认识几个基础词

### Python 类和对象

“类”可以理解为设计图，“对象”是根据设计图造出来的具体机器。

```python
model = VeriStrongDecisionNetwork(hidden_dim=128)
```

`VeriStrongDecisionNetwork` 是类，`model` 是创建出来的对象。

### 函数和方法

普通函数是一段可以调用的代码；写在类里面的函数通常叫方法。

```python
class Example:
    def work(self, value):
        return value + 1
```

这里的 `work` 是 `Example` 类的方法，`self` 代表当前这个具体对象。

### Tensor（张量）

Tensor 可以先理解为“能够放在 CPU 或 GPU 上计算的多维数字表格”。例如：

```text
3 个事务，每个事务有 4 个特征

[[0.1, 1.0, 0.0, 2.0],
 [0.3, 0.0, 1.0, 5.0],
 [0.8, 1.0, 1.0, 3.0]]
```

它的形状是 `(3, 4)`。

### 神经网络模块 `nn.Module`

PyTorch 把一个完整模型和模型内部的小零件都表示成 `nn.Module`。因此，编码器、消息处理器、输出头和完整模型都是模块。

大模块可以包含小模块，就像一台机器可以包含多个零件。

### 图神经网络 GNN

普通神经网络主要处理一排数字；图神经网络处理“节点以及节点之间的边”。

本项目的节点包括：

- `Transaction`：事务；
- `Key`：被读写的数据键；
- `Decision`：还需要模型协助选择的决策；
- `Constraint`：把相关候选组织在一起的约束组。

边表示节点之间的关系。GNN 会让节点沿着边交换信息，这个过程叫“消息传递”。

### WW 和 WR

- `WW`（Write-Write）：两个事务都写了同一个 key，需要确定先后方向；
- `WR`（Write-Read）：一次读取需要确定它读到了哪个写事务的结果。

模型的任务不是直接输出最终证明，而是给这些选择打分。

### logit

`logit` 是模型输出的原始分数。分数越高，一般表示模型越偏向那个候选。它不一定是 0 到 1 之间的概率。

### checkpoint

checkpoint 是保存到磁盘的已训练模型，包括模型参数及相关配置。没有 checkpoint 时，程序不会凭空拥有训练效果。

### witness

witness 是一份完整的 WW/WR 选择结果。只有通过独立校验器后，它才是可接受的结果。

## 3. `forward` 是什么

每个 PyTorch 模块都需要说明：“数据进入我以后，我怎样处理它？”这个处理规则通常写在 `forward` 方法中。

最简单的例子：

```python
class Doubler(nn.Module):
    def forward(self, value):
        return value * 2

module = Doubler()
result = module(3)  # 得到 6
```

虽然代码写的是 `module(3)`，PyTorch 最终会调用这个模块的 `forward`。调用关系可以先简化理解为：

```text
module(3)
   │
   ▼
PyTorch 的 __call__
   │
   ▼
module.forward(3)
```

平时应写 `module(data)`，不要手动写 `module.forward(data)`。前一种方式能让 PyTorch 正常处理 hooks 等模块机制。

## 4. 为什么项目里有多个 `forward`

因为项目里有多个不同模块，每个模块都要定义自己的数据处理方式。

可以把模型想成一家工厂：

- 完整模型的 `forward` 是厂长安排整条流水线；
- 编码器的 `forward` 是原料处理车间的工作流程；
- Processor 的 `forward` 是信息交换车间的一轮工作；
- 输出头的 `forward` 是最终打分车间的工作流程。

它们都叫 `forward`，但属于不同的类，所以不会冲突。类似于“张三有工作方法，李四也有工作方法”，方法都叫“工作”，执行内容却不相同。

当前 VeriStrong 模型中的主要调用链是：

```text
VeriStrongDecisionNetwork.forward
├── transaction_encoder.forward
├── key_encoder.forward
├── decision_encoder.forward
├── constraint_encoder.forward
├── processor.forward × processor_steps
├── ww_head.forward
└── wr_head.forward
```

源码位置：

- `src/isolation_gnn/model.py` 中的 `NodeEncoder.forward`：把原始节点特征编码到统一维度；
- `src/isolation_gnn/model.py` 中的 `RecurrentRelationalProcessor.forward`：完成一轮图消息传递；
- `src/isolation_gnn/veristrong_model.py` 中的 `VeriStrongChoiceHead.forward`：给候选打分；
- `src/isolation_gnn/veristrong_model.py` 中的 `VeriStrongDecisionNetwork.forward`：组织完整前向流程；
- `src/isolation_gnn/veristrong_normalize.py` 中的 `VeriStrongFeatureNormalizer.forward`：标准化输入特征。

所以，“源码中有多个 `forward`”表示有多个不同零件，并不表示完整模型莫名其妙重复运行了多遍。

## 5. 为什么 Processor 的同一个 `forward` 又会执行多次

完整模型里有这样一段循环：

```python
for _ in range(steps):
    states = self.processor(states, graph.relation_edges)
```

假设 `steps=6`，Processor 就会执行 6 轮：

```text
初始节点状态
  → 第 1 轮交换信息
  → 第 2 轮交换信息
  → 第 3 轮交换信息
  → 第 4 轮交换信息
  → 第 5 轮交换信息
  → 第 6 轮交换信息
  → 最终节点状态
```

原因是图中较远的节点不能只靠一轮就知道彼此的信息。多轮传播可以让信息到达更远的位置。

这 6 轮使用的是同一个 `self.processor`，共享同一套参数。项目并没有创建 6 个 Processor。这个设计称为“循环使用”或“参数共享”。

使用默认 `processor_steps=6` 时，一次实际模型推理至少会进入这些自定义 `forward`：

```text
1 次完整模型 forward
4 次节点编码器 forward
6 次 Processor forward
2 次输出头 forward
----------------------
共 13 次模块级 forward
```

如果 checkpoint 还带有标准化器，模型前面会再调用 1 次标准化器的 `forward`。这些调用共同组成“一次完整推理”，不能把它们理解为进行了 13 次完整预测。

## 6. `model.py` 和 `veristrong_model.py` 为什么都像模型

项目保留了两个阶段的实现：

- `model.py` 中的 `IsolationDecisionNetwork` 是较早的 AR/WW 三类节点原型；
- `veristrong_model.py` 中的 `VeriStrongDecisionNetwork` 是当前真实流程使用的 WW/WR 四类节点模型。

当前模型仍然从 `model.py` 复用了两个通用零件：

```python
from .model import NodeEncoder, RecurrentRelationalProcessor
```

所以不要把整个 `model.py` 都当作废代码：

- `NodeEncoder` 和 `RecurrentRelationalProcessor` 当前仍在使用；
- `IsolationDecisionNetwork` 是早期原型和兼容接口；
- 真正组织当前推理的是 `VeriStrongDecisionNetwork`。

刚开始阅读时，可以先跳过旧的 `IsolationDecisionNetwork`，避免同时学习两套数据结构。

## 7. 一次实际推理经过什么

下面这段是项目 README 中 Python API 的核心流程：

```python
history = parse_dbcop_history("history.bincode")
problem = construct_veristrong_problem(history, fast_prune=True)
graph = build_veristrong_graph(problem)

model = VeriStrongDecisionNetwork(hidden_dim=128, processor_steps=6)
model.eval()
with torch.no_grad():
    outputs = model(graph)

ww_scores, wr_scores = scores_from_outputs(graph, outputs)
result = decode_with_beam(problem, ww_scores, wr_scores, beam_size=8)
```

逐行解释：

1. `parse_dbcop_history`：读取二进制历史文件；
2. `construct_veristrong_problem`：根据历史建立待解决的问题，并做确定性剪枝；
3. `build_veristrong_graph`：把问题转换成 GNN 能处理的图；
4. `VeriStrongDecisionNetwork(...)`：创建模型对象；
5. `model.eval()`：切换到推理模式；
6. `torch.no_grad()`：推理时不保存训练所需的梯度，减少内存和计算；
7. `model(graph)`：启动完整模型的 `forward`；
8. `scores_from_outputs`：把模型输出整理成 WW/WR 分数；
9. `decode_with_beam`：根据分数搜索完整 witness，并进行校验。

需要特别注意：如果只是直接创建一个随机模型，却没有加载训练后的 checkpoint，输出只是随机初始化参数产生的分数。

## 8. 训练和推理有什么区别

训练的目的，是根据已有正确答案不断调整模型参数。

```text
输入历史
  → 模型 forward
  → 得到预测
  → 与标签比较，计算 loss
  → backward 计算梯度
  → optimizer 更新参数
```

推理则只使用已经训练好的参数：

```text
输入历史
  → 模型 forward
  → 搜索并校验结果
```

因此：

- `forward` 负责从输入算出预测；
- `loss` 衡量预测和训练标签之间的差距；
- `backward` 根据 loss 计算参数应该怎样调整；
- optimizer 真正更新参数。

这里的 `backward` 和图中的“反向关系边”不是同一个概念。

## 9. 推荐的源码阅读顺序

不要从头到尾阅读所有文件。建议按下面的顺序：

1. `README.md`：先了解项目输入、输出和命令；
2. `src/isolation_gnn/cli.py`：看命令行怎样串起完整流程；
3. `src/isolation_gnn/veristrong_model.py`：看当前模型的整体结构；
4. `src/isolation_gnn/model.py`：只先看 `NodeEncoder` 和 `RecurrentRelationalProcessor`；
5. `src/isolation_gnn/veristrong_graph.py`：理解四类节点如何放进图；
6. `src/isolation_gnn/veristrong_decode.py`：理解模型分数怎样变成最终结果；
7. `src/isolation_gnn/veristrong_loss.py`：开始学习训练时再看；
8. `tests/`：结合小型例子确认自己的理解。

第一次阅读一个函数时，只回答三个问题：

1. 输入是什么？
2. 输出是什么？
3. 中间调用了哪几个更小的模块？

先不要试图理解每一行张量运算。

## 10. 可以安全尝试的命令

运行全部测试：

```bash
pytest
```

查看命令行帮助：

```bash
python3 -m isolation_gnn --help
isolation-gnn-inspect-veristrong --help
isolation-gnn-veristrong --help
```

后两条命令需要先按照 README 完成项目安装。

检查一个真实历史，并让随机初始化模型运行一次前向：

```bash
python3 -m isolation_gnn inspect PATH/TO/history.bincode --forward
```

这里命令行中的 `--forward` 只是一个开关，意思是“除了检查数据，再运行一次模型推理”。它和 Python 类中定义的 `forward` 方法不是同一种东西。

## 11. 新手常见误区

### “多个 `forward` 会互相覆盖吗？”

不会。它们属于不同的类，完整名字分别类似于 `NodeEncoder.forward`、`VeriStrongChoiceHead.forward`。

### “Processor 执行 6 次，就是 6 个模型吗？”

不是。它是同一个模块被循环调用 6 次，而且参数共享。

### “模型给出的最高分一定正确吗？”

不一定。模型只提供搜索顺序，最后必须经过独立校验。

### “`UNKNOWN` 是无解吗？”

不是。它只表示当前搜索预算内没有找到可验证结果。

### “运行 `--forward` 就是在训练吗？”

不是。它只做一次推理，没有计算 loss，也没有更新模型参数。

### “没有 checkpoint 也能得到输出，说明模型训练好了吗？”

不是。随机初始化模型也能输出数字，但这些数字通常没有实际预测意义。

## 12. 第一阶段应该掌握到什么程度

读完后，如果你能用自己的话回答下面五个问题，就已经完成第一阶段：

1. 项目的输入和最终输出分别是什么？
2. 神经网络为什么不是最终正确性的保证？
3. 为什么不同类都可以有一个叫 `forward` 的方法？
4. 为什么同一个 Processor 要循环调用多轮？
5. 当前流程使用的是 `IsolationDecisionNetwork` 还是 `VeriStrongDecisionNetwork`？

答案分别是：输入是数据库历史，输出是经认证的结果状态和可能的 witness；最终结果由确定性校验器保证；不同类有各自独立的方法命名空间；多轮消息传播让更远的节点交换信息；当前流程使用 `VeriStrongDecisionNetwork`。

接下来建议只打开 `src/isolation_gnn/veristrong_model.py`，按照“输入—处理—输出”三个问题逐段阅读。这是理解当前模型最短、最不容易迷路的入口。
