# 实验开发指导手册（学生版）

> **文档定位**：从零开始搭建实验的完整操作指南，覆盖代码环境准备、模型实现、实验运行到结果整理的全流程。
>
> **使用时机**：实验开始前阅读整体思路，各阶段开始前查阅对应章节的具体操作。
>
> **配套文档**：
> - 《实验调试与自检手册》— 实验出问题时的排查指南
> - 《实验多智能体系统设计白皮书》— 老师内部架构文档（学生无需阅读）
>
> **版本**：v1.1

---

## 0. 30秒导航：我现在该翻开哪一页？

```
你现在处于什么状态？
    │
    ├─→ 【还没开始写代码】或【准备进入下一阶段】
    │     └─→ 你正在读正确的文档（本手册）
    │           见下方「按阶段速查表」
    │
    └─→ 【正在跑实验但卡住了/报错了/指标不对】
          └─→ 请切换到《实验调试与自检手册》
                打开它的 §2「问题→智能体快速匹配」
```

### 按阶段速查表

| 如果你现在... | 翻到本手册的... | 你会得到... |
|--------------|----------------|------------|
| 刚拿到论文，不知道实验从哪下手 | §1 论文→实验映射表 | 把挑战/贡献翻译成具体实验的表格 |
| 不知道技术路线怎么搭 | §2-§3 整体思路 + 渐进路线图 | V1→V4开发顺序和各阶段目标 |
| 不知道选哪个基线代码库 | §4 代码库总览 | 方向对应的推荐基线 + 可复用组件 |
| 准备配环境、跑基线 | §5 阶段A | conda命令 + 数据格式 + 复现checklist |
| 开始写第一个模型文件 | §6 阶段B | V1骨架代码模板 + 交付标准 |
| 要加入第一个创新模块 | §7 阶段C | 模块日志模板 + 代码示例 |
| 准备跑完整实验（消融/参数敏感） | §8 阶段D | 14项实验清单 + 消融设计表 + t-test脚本 |
| 实验跑完了，要整理结果写论文 | §9 阶段E | 结果→Abstract映射 + 图表规范 |
| 想用 Claude/DeepSeek 辅助开发 | §10 与AI工具协作 | Prompt模板 + 分工原则 |
| 实验出问题了，不知道查哪个智能体 | §11 诊断智能体使用指南 | 症状→智能体匹配 + 五步法 |

> **更完整的双文档导航** → 参见《实验双文档快速入门》（1页纸）

---

## 目录

1. [前置：从论文到实验的映射表](#一前置从论文到实验的映射表)
2. [整体思路与技术架构](#二整体思路与技术架构)
3. [渐进式开发路线图](#三渐进式开发路线图)
4. [代码库总览与复用指南](#四代码库总览与复用指南)
5. [阶段A：环境搭建与基线复现](#五阶段a环境搭建与基线复现)
6. [阶段B：V1骨架模型搭建](#六阶段bv1骨架模型搭建)
7. [阶段C：逐步加入创新模块](#七阶段c逐步加入创新模块)
8. [阶段D：完整实验设计](#八阶段d完整实验设计)
9. [阶段E：实验结果整理与论文衔接](#九阶段e实验结果整理与论文衔接)
10. [与AI工具的协作方法](#十与ai工具的协作方法)
11. [实验诊断智能体使用指南](#十一实验诊断智能体使用指南)
12. [文件结构与版本管理规范](#十二文件结构与版本管理规范)
13. [附录](#十三附录)

---

## 一、前置：从论文到实验的映射表

### 1.1 为什么需要这张表

论文的Introduction中写的每一个**挑战**和**贡献**，都必须在实验中得到验证。这张表是你的**实验设计锚点**，确保实验不是盲目做，而是有目的地验证论文声明。

### 1.2 映射表模板

在实验开始前，必须填写以下表格（复制到你的实验笔记中）：

```markdown
## 论文→实验映射表

| 论文位置 | 挑战/贡献描述 | 对应实验 | 验证指标 | 预期结果 | 实际结果 | 状态 |
|---------|-------------|---------|---------|---------|---------|------|
| Intro C1 | 早期信号微弱 | 不同观测窗口(10%/25%/50%/75%) | Hits@k | 窗口越小优势越大 | | ☐ |
| Intro C2 | 引导意图隐式 | 消融：w/o GKG | Hits@k | 下降>2% | | ☐ |
| Contrib 1 | GSIB信息瓶颈 | 消融：w/o IB(β=0) | Hits@k+泛化误差 | IB版本更高 | | ☐ |
| Contrib 2 | 连续时间建模 | ODE vs Transformer对比 | Hits@k+训练时间 | 精度相当，更连续 | | ☐ |
| ... | ... | ... | ... | ... | ... | ... |

### 自检规则
- [ ] 每个挑战都有至少一个实验验证
- [ ] 每个贡献都有至少一个实验验证
- [ ] 实验数量 ≥ 挑战数量（理想情况下 1:1 对应）
- [ ] 如果某行填不出来 → 立刻修改论文或补实验
```

### 1.3 与评审标准的对齐

这张表直接对应 Agent 3（Innovation Assessor）的审查项：

| 评审检查项 | 映射表中的对应列 |
|-----------|----------------|
| 贡献-挑战 1:1 对应 | 挑战描述 ↔ 贡献描述 |
| 模块-挑战匹配度 | 对应实验是否能验证挑战 |
| 消融可解释性 | 每个消融是否有明确验证目标 |

> **关键原则**：如果某个挑战/贡献在映射表中找不到对应的实验行，评审时会被判定为"未验证声明"，严重扣分。

---

## 二、整体思路与技术架构

### 2.1 科学问题重述

在开始实验前，用**一句话**重述你的科学问题：

> 例：在引导性话题传播的早期阶段，如何利用信息瓶颈约束和连续时间动力学，实现比现有方法更准确的传播趋势预测？

这句话必须与你的**摘要科学问题句**和**论文题目**保持一致。

### 2.2 技术架构图

画出你的技术架构图，并标注：
- 每个模块的名称和输出
- 模块之间的数据流（维度）
- 与论文贡献的对应关系

```
┌─────────────────────────────────────────┐
│           你的模型名称                    │
│  损失: L = L_main + α*L_aux1 + β*L_aux2  │
└─────────────────────────────────────────┘
              │           │
              ▼           ▼
    ┌─────────────┐ ┌─────────────┐
    │   模块1      │ │   模块2      │
    │  [模块名称]  │ │  [模块名称]  │
    │  输出: [...] │ │  输出: [...] │
    │  →贡献1      │ │  →贡献2      │
    └─────────────┘ └─────────────┘
              │           │
              └─────┬─────┘
                    ▼
            ┌─────────────┐
            │   融合/预测  │
            │  输出: [...] │
            └─────────────┘
```

### 2.3 开发策略：渐进式（不要一次性实现所有创新点）

```
V1（第1-2周）: 基线复现 + 骨架搭建
    → 验证：模块间接口可行（Agent 2 §4.3）
    → 验证：代码能跑通、loss能下降

V2（第3-4周）: 加入核心创新模块1
    → 验证：模块-挑战匹配度 ≥ B级（Agent 3 §4.2）
    → 验证：指标比V1有提升

V3（第5-6周）: 加入核心创新模块2（如有）
    → 验证：完整模型指标优于V2
    → 验证：消融实验可解释

V4（第7-8周）: 完整实验 + 消融 + 参数敏感
    → 验证：通过 Agent 2 的14项实验清单
    → 验证：通过 Agent 6 的一票否决项自检
```

**关键原则**：每个版本都是**可运行、可评估**的独立模型。不要试图一次性实现所有创新点。

---

## 三、渐进式开发路线图

### 3.1 总览时间线

```
Week 1-2          Week 3-4          Week 5-6          Week 7-8
   │                 │                 │                 │
   ▼                 ▼                 ▼                 ▼
┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
│ 阶段A  │  →   │ 阶段B  │  →   │ 阶段C  │  →   │ 阶段D  │
│ 环境+  │      │ V1骨架 │      │ V2+V3  │      │ 完整   │
│ 基线   │      │ 搭建   │      │ 创新   │      │ 实验   │
└────────┘      └────────┘      └────────┘      └────────┘
   │                 │                 │                 │
   ▼                 ▼                 ▼                 ▼
交付物：         交付物：         交付物：         交付物：
- 环境就绪       - V1能跑通       - 完整模型       - 主实验表
- 基线指标对齐   - 指标>随机      - 指标>基线      - 消融表
- 数据加载验证   - 梯度正常回传   - 消融设计完成   - 参数敏感图
```

### 3.2 各阶段里程碑

| 阶段 | 里程碑 | 不通过标准 |
|------|--------|-----------|
| A | 基线指标复现误差 < 2% | 误差 > 5% → 排查数据/代码（见调试手册§3） |
| B | V1能跑完100 epoch，loss下降 | Loss不下降 → 调试手册§4/§5 |
| C | 完整模型指标 > 最强基线 | 未超过 → 检查模块实现/调参 |
| D | 消融+参数敏感+统计显著性全部完成 | 缺少任一 → 补实验 |

---

## 四、代码库总览与复用指南

### 4.1 选择基线代码库

根据你的方向，从以下推荐基线中选择1个主代码库：

| 方向 | 推荐基线 | 代码链接 | 推荐理由 | 适用场景 |
|------|---------|---------|---------|---------|
| 传播预测 | BuzzBloom | GitHub | 多模型统一框架，组件可复用 | 级联预测、超图、时序 |
| 传播预测 | CasFlow | GitHub | TKDE标杆，文档清晰 | 级联预测、层次结构 |
| 谣言检测 | KPG | GitHub | SOTA性能，含Weibo22数据 | 图神经网络谣言检测 |
| 谣言检测 | RAGCL/GACL | GitHub | 对比学习模块完整 | 图对比学习 |
| 社交机器人 | UnDBot/BotCGP | GitHub | 唯二开源代码 | 结构熵、群体感知 |

> **选择原则**：选与你技术路线最接近、代码结构最清晰的基线，不要贪多。

### 4.2 目录结构与可复用组件

以 **BuzzBloom** 为例（其他基线结构类似）：

```
Baseline_Code/
├── run.py                      # 主入口（通常无需修改）
├── requirements.txt            # 依赖清单
├── data/                       # 数据集目录
│   └── {dataset_name}/
│       ├── cascades.txt        # 级联数据
│       ├── edges.txt           # 社交关系
│       └── ...
├── models/                     # 模型定义（你主要修改这里）
│   ├── BaselineModel1.py       # 基线1
│   ├── BaselineModel2.py       # 基线2
│   └── ...
├── layers/                     # 公共组件层（可复用）
│   ├── Commons.py              # GNN层、注意力、融合
│   ├── GraphBuilder.py         # 图构建工具
│   └── TransformerBlock.py     # Transformer模块
├── helpers/                    # 训练/评估框架（通常直接继承）
│   ├── BaseLoader.py           # 数据加载器基类
│   └── BaseRunner.py           # 训练循环基类
└── utils/                      # 工具函数
    ├── Metrics.py              # 评估指标
    └── Optim.py                # 优化器配置
```

### 4.3 可复用组件清单

填写你选择的基线中，哪些组件可以直接复用：

| 组件 | 文件位置 | 在你的方案中的角色 | 需要修改？ |
|------|---------|------------------|-----------|
| BaseLoader | helpers/BaseLoader.py | 数据加载 | ☐ 否 / ☐ 是（说明：___） |
| BaseRunner | helpers/BaseRunner.py | 训练循环 | ☐ 否 / ☐ 是（说明：___） |
| Metrics | utils/Metrics.py | 评估指标 | ☐ 否 / ☐ 是（说明：___） |
| GraphBuilder | layers/GraphBuilder.py | 图构建 | ☐ 否 / ☐ 是（说明：___） |
| GNN Layer | layers/Commons.py | 图编码 | ☐ 否 / ☐ 是（说明：___） |
| Attention | layers/Commons.py | 注意力 | ☐ 否 / ☐ 是（说明：___） |
| Transformer | layers/TransformerBlock.py | 序列编码 | ☐ 否 / ☐ 是（说明：___） |

### 4.4 基线复现命令与指标对齐

在实验开始前，必须完成以下基线复现表：

| 基线 | 复现命令 | 论文报告指标 | 你的复现指标 | 差异 | 是否通过 |
|------|---------|------------|------------|------|---------|
| Baseline1 | `python run.py --model_name X ...` | Hits@10=0.XX | | < 2% | ☐ |
| Baseline2 | `python run.py --model_name Y ...` | Hits@10=0.XX | | < 2% | ☐ |
| ... | ... | ... | ... | ... | ... |

> **不通过标准**：差异 > 5% 时，必须排查数据加载或评估代码实现是否有误。参考《实验调试与自检手册》§3（DataAgent）。

---

## 五、阶段A：环境搭建与基线复现

### 5.1 环境搭建（0.5-1天）

```bash
# Step 1: 创建虚拟环境
conda create -n your_project python=3.10
conda activate your_project

# Step 2: 安装基线依赖
cd /path/to/baseline_code
pip install -r requirements.txt

# Step 3: 验证GPU可用
python -c "import torch; print(torch.cuda.is_available())"

# Step 4: 运行基线测试（确认环境无误）
python run.py --model_name Baseline1 --data_name your_dataset --epoch 5
```

### 5.2 数据准备（1天）

#### 数据格式要求（以级联预测为例）

```
# data/{your_dataset}/cascades.txt
# 每行一条级联，格式：user_id timestamp,user_id timestamp,...
10743 1.0,10074 4.0,10727 7.0,...

# data/{your_dataset}/edges.txt
# 每行一条社交关系边：user1,user2
10743,10074
10074,10727
```

#### 数据集统计表（必须填写，对应 Agent 2 E2）

| 数据集 | 用户数 | 边数 | 级联数 | 平均级联长度 | 特征维度 | 任务类型 |
|--------|--------|------|--------|------------|---------|---------|
| Dataset A | | | | | | |
| Dataset B | | | | | | |

### 5.3 基线复现（2-3天）

#### 操作清单

| 任务 | 预估时间 | 说明 | 交付标准 |
|------|---------|------|---------|
| A1. 复现基线1 | 0.5天 | 运行官方命令 | 指标误差 < 2% |
| A2. 复现基线2 | 0.5天 | 运行官方命令 | 指标误差 < 2% |
| A3. 复现基线3 | 0.5天 | 运行官方命令 | 指标误差 < 2% |
| A4. 记录指标对比 | 0.5天 | 整理表格 | 表格完整 |

#### 交付标准
- 所有基线指标能稳定复现（误差 < 2%）
- 同一个GPU下一个epoch训练时间 < 30s（如差距大，检查数据加载效率）
- 训练日志已保存到 `log/` 目录

---

## 六、阶段B：V1骨架模型搭建

### 6.1 V1目标

用基线代码库中的**现有组件**搭一个简化版模型，验证：
1. 多模块架构能协同工作
2. 代码能跑通、loss能下降
3. 各模块梯度能正常回传

### 6.2 V1结构设计

```markdown
## MyModelV1 结构

模块1：[基线组件A]  → 解决：[简化版挑战1]
模块2：[基线组件B]  → 解决：[简化版挑战2]
模块3：[基线组件C]  → 解决：[融合/预测]

创新点：V1**不加入**任何创新模块，仅验证架构可行性
```

### 6.3 创建模型文件

**文件位置**：`models/MyModelV1.py`

**必须实现的接口**：

```python
class MyModelV1(nn.Module):
    """
    接口规范（对应 writing-standards §4.3 + §5.2）：
    
    输入:
      - input_seq: [batch_size, seq_len] 用户ID序列
      - input_timestamp: [batch_size, seq_len] 时间戳序列
    
    输出:
      - pred_logits: [batch_size * seq_len, n_users] 下一个用户预测logits
    
    模块间数据流:
      - module1_output: [n_users, hidden_size] → module2_input
      - module2_output: [batch, seq, hidden] → fusion_input
    """
    
    # 必须声明Loader和Runner（基线框架要求）
    Loader = BaseLoader
    Runner = BaseRunner
    
    @staticmethod
    def parse_model_args(parser):
        """添加模型专属超参数"""
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--time_step_split', type=int, default=8)
        return parser
    
    def __init__(self, args, data_loader):
        super().__init__()
        # 模型初始化...
        
    def forward(self, input_seq, input_timestamp, tgt_idx):
        # 前向计算...
        return output
    
    def get_performance(self, input_seq, input_timestamp, history_idx, 
                        loss_func, gold):
        # 损失计算...
        return loss, n_correct
```

### 6.4 代码骨架示例

```python
# models/MyModelV1.py
import torch
import torch.nn as nn
from layers.Commons import GraphNN, HierarchicalGNNWithAttention, Fusion
from layers.TransformerBlock import TransformerBlock
from layers import GraphBuilder
from utils import Constants
from helpers.BaseLoader import BaseLoader
from helpers.BaseRunner import BaseRunner

class MyModelV1(nn.Module):
    Loader = BaseLoader
    Runner = BaseRunner

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--time_step_split', type=int, default=8)
        return parser

    def __init__(self, args, data_loader):
        super().__init__()
        self.device = args.device
        self.hidden_size = args.d_model
        self.n_heads = args.n_heads
        self.n_node = data_loader.user_num

        # 模块1：静态图编码（复用基线组件）
        self.social_graph = GraphBuilder.build_friendship_network(data_loader)
        self.fri_gnn = GraphNN(self.n_node, self.hidden_size)

        # 模块2：动态图编码（复用基线组件）
        self.hyper_graph_list = GraphBuilder.build_diff_hyper_graph_list(
            data_loader.cascades, data_loader.timestamps, self.n_node
        )
        self.diff_gnn = HierarchicalGNNWithAttention(
            self.hidden_size, self.hidden_size * 2, self.hidden_size
        )

        # 模块3：融合与预测（复用基线组件）
        self.fusion = Fusion(self.hidden_size)
        self.transformer = TransformerBlock(
            input_size=self.hidden_size, n_heads=self.n_heads
        )
        self.linear = nn.Linear(self.hidden_size, self.n_node)
        self.user_embedding = nn.Embedding(self.n_node, self.hidden_size, 
                                           padding_idx=0)

    def forward(self, input_seq, input_timestamp, tgt_idx):
        # 静态图编码
        static_emb = self.fri_gnn(self.social_graph)
        
        # 动态图编码
        dynamic_emb_list = self.diff_gnn(static_emb, self.hyper_graph_list)
        
        # 序列嵌入
        batch_size, seq_len = input_seq.shape
        seq_emb = self.user_embedding(input_seq)
        
        # 融合与预测
        fused = self.fusion(seq_emb, seq_emb)
        output = self.transformer(fused, fused, fused)
        output = self.linear(output)
        
        return output.view(-1, output.size(-1))

    def get_performance(self, input_seq, input_timestamp, history_idx, 
                        loss_func, gold):
        pred = self.forward(input_seq, input_timestamp, history_idx)
        loss = loss_func(pred, gold.contiguous().view(-1))
        pred_labels = pred.max(1)[1]
        gold_flat = gold.contiguous().view(-1)
        non_pad_mask = gold_flat.ne(Constants.PAD)
        n_correct = pred_labels.eq(gold_flat).masked_select(non_pad_mask).sum().float()
        return loss, n_correct
```

### 6.5 运行与验证

```bash
# 运行V1
python run.py --model_name MyModelV1 --data_name your_dataset \
              --epoch 100 --batch_size 2048 --d_model 64

# 快速调试模式（仅5 epoch，小模型）
python run.py --model_name MyModelV1 --data_name your_dataset \
              --epoch 5 --d_model 16 --batch_size 128
```

### 6.6 V1交付标准

- [ ] V1能跑完100个epoch不报错
- [ ] Loss曲线呈下降趋势（前10epoch即可看出）
- [ ] 各模块梯度都能正常回传（打印grad_norm验证）
- [ ] 指标高于随机水平
- [ ] 每个epoch训练时间合理（与基线同量级）

> 如果V1无法通过上述标准，**不要进入V2**，先参考《实验调试与自检手册》排查问题。

---

## 七、阶段C：逐步加入创新模块

### 7.1 每个模块加入时的必填信息

每加入一个创新模块，必须填写以下信息（创建 `module_log.md`）：

```markdown
## 模块X：{模块名称}

### 基础信息
- **加入版本**：V2 / V3 / V4
- **解决的挑战**：{对应 Introduction 中的挑战X}
- **对应的贡献**：{对应论文中的贡献X}

### 实现规格
- **输入**: {张量形状 + 物理含义}
- **输出**: {张量形状 + 物理含义}
- **新增参数**: {列表}
- **参考代码**: {复用了哪个开源仓库的哪个文件}

### 模块-挑战匹配度自评
- [ ] A级：直接解决挑战核心矛盾
- [ ] B级：间接缓解挑战
- [ ] C级：关联模糊（需警惕）
- [ ] D级：无法解决（必须重构）

### 消融验证设计
- **消融变体名称**：w/o {模块名}
- **验证的因果命题**："{模块名}能通过{机制}解决{挑战}"
- **预期结果**：去掉后{指标}下降{X}%
- **实际结果**：{待填写}

### 接口检查
- [ ] 形式接口：输出维度 = 下游模块输入维度
- [ ] 语义接口：输出物理含义与下游需求一致
- [ ] 梯度接口：连接操作可导
```

### 7.2 V2示例：加入信息瓶颈约束

#### 设计思路

```
标准信息瓶颈: L = I(Z; Y) - beta * I(Z; X)
你的实现:      L = CE_pred + beta * KL(Z || prior)
```

#### 核心代码

```python
class MyModelV2(nn.Module):
    def __init__(self, args, data_loader):
        # ... 继承V1的所有内容 ...
        
        # 新增：变分编码器（用于IB约束）
        self.encoder_mu = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_logvar = nn.Linear(self.hidden_size, self.hidden_size)
        self.beta = getattr(args, 'beta', 1.0)  # IB约束系数
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, input_seq, input_timestamp, tgt_idx):
        # ... V1的前向计算 ...
        
        # 新增：变分编码
        mu = self.encoder_mu(fused_repr)
        logvar = self.encoder_logvar(fused_repr)
        z = self.reparameterize(mu, logvar)
        
        # 用z替代fused_repr进行后续预测
        output = self.transformer(z, z, z)
        output = self.linear(output)
        
        return output.view(-1, output.size(-1)), mu, logvar
    
    def get_performance(self, input_seq, input_timestamp, history_idx, 
                        loss_func, gold):
        pred, mu, logvar = self.forward(input_seq, input_timestamp, history_idx)
        
        # 预测损失
        pred_loss = loss_func(pred, gold.contiguous().view(-1))
        
        # KL散度（信息瓶颈约束）
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= gold.size(0)
        
        # 总损失
        total_loss = pred_loss + self.beta * kl_loss
        
        # 计算准确率（与V1一致）
        pred_labels = pred.max(1)[1]
        gold_flat = gold.contiguous().view(-1)
        non_pad_mask = gold_flat.ne(Constants.PAD)
        n_correct = pred_labels.eq(gold_flat).masked_select(non_pad_mask).sum().float()
        
        return total_loss, n_correct
```

#### 新增超参数

```bash
python run.py --model_name MyModelV2 --data_name your_dataset \
              --beta 0.01 --d_model 64 --epoch 100
```

#### V2交付标准

- [ ] V2能收敛且loss曲线正常（包含KL项）
- [ ] 消融实验：w/o IB（beta=0）vs w/ IB（beta=最优值），IB版本指标更高
- [ ] 至少测试3个beta值（0.001, 0.01, 0.1），找到最优窗口

### 7.3 渐进式开发检查点

| 版本 | 检查点 | 未通过则 |
|------|--------|---------|
| V1 | 骨架能跑通 | 排查接口/维度（调试手册§4） |
| V2 | 模块1有效 | 检查模块-挑战匹配度（调试手册§4.2 M9） |
| V3 | 模块2有效 | 检查模块间耦合（调试手册§4.2 M10） |
| V4 | 完整模型 > 基线 | 全面排查（调试手册全流程） |

---

## 八、阶段D：完整实验设计

### 8.1 主实验设计（对应 Agent 2 §6.2 14项清单）

#### 实验设计表

| 检查项 | 实验设计 | 实现方式 | 状态 |
|--------|---------|---------|------|
| **E1** 数据集≥2 | 数据集A, 数据集B | 准备多个数据集 | ☐ |
| **E2** 统计特征表 | 节点数/边数/特征维度 | 写脚本统计 | ☐ |
| **E3** 基线3–5 | Baseline1-5 | run.py切换 | ☐ |
| **E4** 近3年SOTA | 查知识库补最新基线 | 阅读相关论文 | ☐ |
| **E5** 指标匹配任务 | Hits@k/MAP@k（预测任务） | BaseRunner.test_epoch() | ☐ |
| **E6** 统计显著性 | 5个种子 + t-test | 写脚本 `run_significance.sh` | ☐ |
| **E7** 标准差 | 表格加 ±std | 修改结果输出 | ☐ |
| **E8** 消融实验 | 每个核心模块逐一移除 | 创建消融变体 | ☐ |
| **E9** 参数敏感性 | 至少2个关键超参数 | 网格搜索/随机搜索 | ☐ |
| **E10** AUC/PR曲线 | 提供曲线图 | matplotlib绘制 | ☐ |
| **E11** early检测验证 | 10%/25%/50%/75%比例 | 修改数据加载器 | ☐ |
| **E12** 可视化 | 注意力/embedding可视化 | 写可视化脚本 | ☐ |
| **E13** 计算效率 | 训练/推理时间对比 | time模块计时 | ☐ |
| **E14** 实现可复现 | 框架/硬件/超参数明确 | 写README | ☐ |

#### 主实验结果表（Table 2风格）

| Model | Dataset A | Dataset B | ... |
|-------|-----------|-----------|-----|
| | Hits@10 | MAP@10 | Hits@10 | MAP@10 |
| Baseline1 | XX.X±X.X | XX.X±X.X | XX.X±X.X | XX.X±X.X |
| Baseline2 | XX.X±X.X | XX.X±X.X | XX.X±X.X | XX.X±X.X |
| **Ours** | **XX.X±X.X** | **XX.X±X.X** | **XX.X±X.X** | **XX.X±X.X** |
| Improv. | +X.X% | +X.X% | +X.X% | +X.X% |

> 格式要求：最佳结果**加粗**，次佳结果*斜体*，必须包含 ±std。

### 8.2 消融实验设计（对应 Agent 3 §4.5）

#### 设计原则

每条消融必须验证一个**可证伪的因果命题**，而非简单的"模块重要性投票"。

#### 消融实验表

| 变体 | 操作 | 验证的因果命题 | 预期结果 | 与贡献/挑战对应 | 实际结果 |
|------|------|--------------|---------|----------------|---------|
| w/o A | 移除模块A | "模块A通过XX机制解决YY挑战" | 下降X% | 贡献1 / 挑战1 | |
| w/o B | 移除模块B | "模块B通过XX机制解决YY挑战" | 下降X% | 贡献2 / 挑战2 | |
| w/o A+B | 同时移除 | "A和B存在协同效应" | 下降 > 单独移除之和 | — | |
| 替代方案 | 用GCN替代GAT | "GAT的注意力机制优于GCN" | 下降或持平 | — | |

### 8.3 参数敏感性分析

| 参数 | 测试范围 | 分析目标 | 图表类型 |
|------|---------|---------|---------|
| β (IB系数) | {0, 0.0001, 0.001, 0.01, 0.1, 1.0} | 找到最优窗口，证明非单调 | 折线图 |
| d_model | {32, 64, 128, 256} | 容量与过拟合 | 折线图 |
| batch_size | {512, 1024, 2048, 4096} | 训练效率与稳定性 | 柱状图 |
| n_heads | {2, 4, 8} | 注意力头数影响 | 折线图 |

### 8.4 Early Detection验证（如适用，E11 P0一票否决）

如果Introduction提到"early detection"，必须做：

| 观测比例 | 10% | 25% | 50% | 75% | 100% |
|---------|-----|-----|-----|-----|------|
| MyModel | | | | | |
| Baseline1 | | | | | |
| Baseline2 | | | | | |
| 优势幅度 | | | | | |

> **关键判定**：在10%和25%窗口，你的模型优势应该最明显。如果不是，说明"early detection"声明不成立。

### 8.5 统计显著性检验脚本模板

```bash
#!/bin/bash
# run_significance.sh

MODEL="MyModelV2"
DATA="your_dataset"
SEEDS=(42 123 456 789 1024)

for SEED in "${SEEDS[@]}"; do
    python run.py --model_name $MODEL --data_name $DATA \
                  --seed $SEED --epoch 100 --d_model 64
done

# 然后用Python脚本做t-test
python t_test.py --results_dir results/$MODEL/$DATA/
```

```python
# t_test.py
import numpy as np
from scipy import stats

# 读取5个种子的结果
your_results = [0.85, 0.86, 0.85, 0.87, 0.86]  # 示例
baseline_results = [0.82, 0.83, 0.81, 0.82, 0.83]  # 示例

# paired t-test
t_stat, p_value = stats.ttest_rel(your_results, baseline_results)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant (p<0.05): {p_value < 0.05}")
```

---

## 九、阶段E：实验结果整理与论文衔接

### 9.1 实验结果 → Abstract 结果段

| 实验 | 关键数字 | 填入Abstract位置 |
|------|---------|-----------------|
| 主实验 | 比SOTA提升 X% | E5 结果量化 |
| 消融 | w/o X 下降 Y% | 方法有效性佐证 |
| 早期检测 | 10%窗口提升 Z% | 核心优势 |

### 9.2 结果整理清单

```markdown
□ 所有表格已按期刊格式排版（三线表、加粗最佳结果）
□ 所有图表已导出为高分辨率PDF/PNG（≥300dpi）
□ 实验日志已备份（包括失败的尝试）
□ 模型checkpoint已保存（用于复现）
□ 超参数配置已记录（JSON/YAML格式）
□ 随机种子已记录
□ 训练时间已记录
```

### 9.3 与论文写作的交叉引用

| 论文章节 | 需要引用的实验结果 | 对应本手册章节 |
|---------|-----------------|-------------|
| Abstract结果段 | 主实验最佳指标 | §8.1 主实验表 |
| Method模块说明 | 模块输入输出定义 | §7.1 模块日志 |
| Experiment主实验 | 完整对比表 | §8.1 Table 2 |
| Experiment消融 | 消融表+因果命题 | §8.2 消融设计 |
| Experiment参数敏感 | 参数曲线图 | §8.3 参数表 |

---

## 十、与AI工具的协作方法

### 10.1 工具分工表

| 工具 | 最佳场景 | 输入 | 输出 | 注意事项 |
|------|---------|------|------|---------|
| **Claude Code** | 代码实现、调试、重构 | 代码文件 + 自然语言需求 | 修改后的代码 | 改后必须运行验证 |
| **Kimi/DeepSeek** | 算法设计、公式推导、方案审计 | 论文段落 + 问题 | 伪代码/推导/清单 | 不直接修改代码 |
| **GPT** | 论文写作润色 | Method/Abstract草稿 | 润色后文本 | 不修改技术内容 |

### 10.2 Claude Code 专用Prompt模板

#### 模板1：创建新模块

```markdown
我在基于 {基线名称} 代码库开发一个 {方向} 模型。

当前任务：在 {现有文件} 基础上创建 {新文件}，加入 {模块描述}。

上下文：
- 基线代码位置：{文件路径}
- 当前模型输入输出：{input [shape] → output [shape]}
- 新增需求：{自然语言描述}
- 新增超参数：{列表}

约束：
1. 不要修改 BaseLoader 和 BaseRunner
2. 保持 forward 的输出接口不变
3. get_performance 返回 (total_loss, n_correct)
4. 增加注释说明输入输出维度

请生成完整代码并解释关键修改点。
```

#### 模板2：调试维度不匹配

```markdown
我遇到了 RuntimeError: size mismatch 错误。

错误信息：
```
{粘贴完整报错信息}
```

相关代码：
```python
{粘贴相关代码片段}
```

期望的维度流：
- 输入：{shape}
- 模块A输出：{shape}
- 模块B输入：{shape}
- 最终输出：{shape}

请帮我定位问题并给出修复代码。
```

#### 模板3：添加消融变体

```markdown
我需要在 {模型文件} 中创建一个消融变体 "w/o {模块名}"。

要求：
1. 创建 {新类名} 类，继承自 {原类名}
2. 在 __init__ 中移除 {模块名}，但保持系统能运行
3. 在 forward 中跳过 {模块名} 的调用
4. 确保输出维度与原始模型一致（用于公平对比）

注意：
- 移除该模块后，可能需要调整下游模块的输入
- 保持其他所有代码不变

请生成完整代码。
```

### 10.3 使用AI工具的原则

1. **一次一个任务**：不要把10个修改放在同一个Prompt里
2. **改前commit**：任何AI修改前先用git保存当前状态
3. **改后验证**：AI给的代码必须先跑5个epoch快速验证
4. **保留证据**：把AI生成的关键代码片段保存到 `ai_generated/` 目录，便于论文中的方法描述

---

## 十一、实验诊断智能体使用指南

> 本章节将《实验调试与自检手册》中的 **4+1 诊断智能体体系**（DataAgent / ModelAgent / TrainAgent / AblateAgent / Meta-Diagnoser）融入实验开发流程，作为学生遇到问题时的系统化排查导航。

### 11.1 诊断智能体与AI编程工具的分工

| 角色 | 职责 | 具体任务 | 禁止行为 |
|------|------|---------|---------|
| **DataAgent** | 数据质量诊断 | 检查数据格式、泄漏、划分、PAD处理 | 不修改模型架构 |
| **ModelAgent** | 模型结构诊断 | 检查维度、梯度、接口、模块-挑战匹配度 | 不调超参数 |
| **TrainAgent** | 训练动态诊断 | 检查lr、优化器、过拟合、数值稳定性 | 不改数据预处理 |
| **AblateAgent** | 消融实验诊断 | 检查消融设计、因果命题、对照组完整性 | 不诊断训练bug |
| **Claude Code** | 代码实现 | 写修复代码、创建消融变体、重构 | 不定位问题根因 |
| **DeepSeek/Kimi** | 算法设计 | 解释原理、设计实验方案、审计公式 | 不直接修改代码 |

**关键原则：先诊断，后编码**

```
❌ 错误：Loss不下降 → 直接问Claude Code "帮我改代码" → 乱改 → 更乱
✅ 正确：Loss不下降 → TrainAgent检查T1-T3 → 发现lr太大 → 让Claude精确修改lr
```

### 11.2 各开发阶段对应的智能体速查表

| 开发阶段 | 典型症状 | 首选智能体 | 次选智能体 | 调试手册章节 | 预计时间 |
|---------|---------|-----------|-----------|------------|----------|
| V1骨架搭建 | RuntimeError / size mismatch | ModelAgent | DataAgent | §4.2 M1-M2 | 10-30min |
| V1指标异常低 | 接近随机或接近0 | DataAgent | ModelAgent | §3.2 D1-D6 | 20min |
| V2加入模块后Loss爆炸 | NaN / 极大数值 | TrainAgent | ModelAgent | §5.2 T1-T3 | 20min |
| V2消融反直觉 | w/o X 性能反而提升 | AblateAgent | ModelAgent | §6.3 | 30-60min |
| V3训练极慢 | 1epoch时间突增 | TrainAgent | — | §5.2 T4-T8 | 15min |
| V3Loss震荡 | 曲线锯齿状 | TrainAgent | ModelAgent | §5.2 T1-T4 | 20min |
| 基线复现误差>5% | 与论文报告差距大 | DataAgent | — | §3.2 D1-D10 | 30-60min |
| 消融缺少对照组 | 无法解释结果 | AblateAgent | — | §6.2 A1-A7 | 20min |

> **建议**：将此表打印贴在工位，遇到问题30秒内定位排查方向。

### 11.3 智能体调用五步法

#### Step 1: 收集证据（2分钟）

```markdown
□ 复制最近完整报错信息（如有RuntimeError）
□ 截取当前loss曲线（最近20个epoch）
□ 记录当前超参数（lr, batch_size, 模型专属参数）
□ 记录当前指标 vs 基线指标
□ 记录当前代码git commit号（方便回滚）
```

#### Step 2: 症状匹配（1分钟）

对照上方速查表，确定：
1. **首选智能体**（主排查方向）
2. **次选智能体**（首选未解决时转向）
3. **优先级**（P0=立即处理，P1=今天处理，P2=有空处理）

**元调度器核心规则（必须遵守）**：

```
数据问题（DataAgent）> 模型问题（ModelAgent）> 训练问题（TrainAgent）> 消融问题（AblateAgent）
```

低层问题会掩盖高层问题。若同时怀疑数据泄漏、模型bug、训练参数不对，**必须先查数据**。

#### Step 3: 执行检查清单（5-15分钟）

打开《实验调试与自检手册》，找到对应智能体的检查清单，**逐项勾选**：

| 智能体 | 必查项 | 关键检查项 |
|--------|--------|-----------|
| DataAgent | D1-D10 | D5（数据泄漏）、D6（PAD处理） |
| ModelAgent | M1-M10 | M2（维度匹配）、M6（梯度正常）、M9（模块-挑战匹配） |
| TrainAgent | T1-T10 | T1（学习率）、T3（调度器）、T6（过拟合） |
| AblateAgent | A1-A8 | A1（消融语义）、A4（下降幅度合理）、A8（统计显著性） |

**执行原则**：
- 不跳过任何一项，即使觉得"这个肯定没问题"
- 每项给出明确的"通过/不通过"结论
- 不通过项立即记录，作为Step 4的修复目标

#### Step 4: 修复并验证（10-30分钟）

```markdown
□ 修复前先git commit
□ 一次只改一处（不要同时改lr和batch_size）
□ 修复后运行5个epoch快速验证（不要跑完整实验）
□ 记录修复动作和验证结果
```

**验证通过标准**：
- 修复后5个epoch内loss趋势正常
- 如果是维度/数据问题，训练不再报错
- 如果是超参数问题，loss曲线明显改善

#### Step 5: 升级至元调度器

当以下情况发生时，启动**元调度器**（Meta-Diagnoser）：

1. 按首选+次选智能体清单全部排查后，问题仍未解决
2. 两个智能体建议相互矛盾（如ModelAgent说"增加容量"，TrainAgent说"过拟合"）
3. 问题涉及多个层面（数据+模型+训练同时异常）
4. 连续3轮修复-验证循环后问题依然存在

**升级操作**：携带完整的检查清单结果 + 所有尝试过的修复方案及效果，找老师/助教进行人工诊断。

### 11.4 元调度器使用示例

#### 示例：消融反直觉（AblateAgent + ModelAgent 冲突）

**场景**：w/o 模块A 后性能反而提升。

- AblateAgent结论：模块A可能冗余或有害
- ModelAgent结论：模块A与挑战匹配度A级，不应有害
- **元调度器裁决**：标记为"设计级矛盾"，需要人工判断
- **可能原因**：模块A超参数过强导致过拟合；或模块A与另一模块B冗余
- **行动**：
  1. 单独测试模块A在不同超参数下的效果
  2. 增加"同时去掉A和B"的消融变体
  3. 若仍矛盾，与老师讨论是否修改论文贡献声明

#### 示例：训练Loss震荡（TrainAgent 建议冲突）

**场景**：Loss震荡，TrainAgent T1建议"降低lr"，T4建议"增大batch_size"。

- **元调度器裁决**：T1（学习率）优先级高于T4（batch_size），先执行降低lr
- **原因**：学习率直接影响优化器步长，是更根本的参数
- **行动**：
  1. 先将lr减半，观察5个epoch
  2. 若仍震荡，再增大batch_size
  3. 不要同时执行两个修改

### 11.5 每日/每周自检清单

**每日实验前（2分钟）**：
```markdown
□ 固定随机种子
□ 确认train/eval模式切换正确
□ 确认学习率、batch_size与计划一致
□ 确认保存路径不会覆盖已有结果
□ 确认上一版本检查清单已全部通过
```

**每周实验后（10分钟）**：
```markdown
□ 基线指标复现误差 < 2%
□ 训练日志完整保存
□ 至少完成一个消融变体
□ 超参数配置已记录
□ 实验结果已填入论文→实验映射表
□ 本周问题及修复方案已记录
```

---

## 十二、文件结构与版本管理规范

### 11.1 推荐项目结构

```
your_project/
├── models/
│   ├── MyModelV1.py          # 阶段B：简化骨架
│   ├── MyModelV2.py          # 阶段C1：+ 创新模块1
│   ├── MyModelV3.py          # 阶段C2：+ 创新模块2
│   └── MyModelV4.py          # 阶段C3：+ 创新模块3（可选）
├── experiment_scripts/        # 实验脚本
│   ├── run_baselines.sh      # 运行所有基线
│   ├── run_ablation.sh       # 运行所有消融
│   ├── run_hyperopt.sh       # 超参数搜索
│   └── run_significance.sh   # 统计显著性检验
├── notebooks/                 # 分析笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_result_analysis.ipynb
│   └── 03_ablation_study.ipynb
├── results/                   # 实验结果
│   ├── baseline_comparison.csv
│   ├── ablation_results.csv
│   └── hyperparameter_sensitivity.csv
├── ai_generated/              # AI生成的代码片段（存档）
│   └── {date}_{task}.py
├── module_logs/               # 模块设计日志
│   ├── module1_vae.md
│   └── module2_ode.md
├── log/                       # 训练日志（自动生成）
├── saved_models/              # 模型检查点（自动生成）
├── README.md                  # 项目说明（复现指南）
└── requirements.txt           # 依赖清单
```

### 11.2 Git版本管理规范

```bash
# 每次重要进展后commit
git add .
git commit -m "V1骨架完成，能跑通100epoch"

# 每次AI修改前commit
git add .
git commit -m "before AI: 加入信息瓶颈模块"

# 标签标记里程碑
git tag -a v1-baseline -m "基线复现完成"
git tag -a v2-ib -m "信息瓶颈模块加入完成"
```

### 11.3 实验记录规范

每次实验运行后，记录以下信息到 `experiments_log.md`：

```markdown
## 实验记录 #{编号}

- **日期**：2026-XX-XX
- **版本**：V2
- **分支**：main
- **commit**：abc1234
- **目的**：验证beta=0.01时IB约束有效性
- **超参数**：--beta 0.01 --d_model 64 --epoch 100
- **结果**：Hits@10=0.852±0.003
- **对比**：beta=0时 Hits@10=0.841±0.004
- **结论**：beta=0.01有效，提升1.1%
- **下一步**：测试beta=0.1
```

---

## 十二、附录

### 附录A：投稿前实验自查表（对应 Agent 2 + Agent 6）

| # | 检查项 | 状态 | 未通过则 |
|---|--------|------|---------|
| E1 | 数据集 ≥2 个 | ☐ | 补充数据集 |
| E2 | 数据集统计表已填 | ☐ | 写脚本统计 |
| E3 | 基线 3–5 个 | ☐ | 补充基线 |
| E4 | 近3年SOTA已覆盖 | ☐ | 查知识库补基线 |
| E5 | 指标与任务匹配 | ☐ | 检查指标定义 |
| E6 | 5个种子 + t-test | ☐ | 跑多次实验 |
| E7 | 标准差/置信区间 | ☐ | 修改结果输出 |
| E8 | 消融 ≥4 项且有语义 | ☐ | 补充消融 |
| E9 | 参数敏感性 ≥2个参数 | ☐ | 补充实验 |
| E10 | AUC/PR曲线（如适用） | ☐ | 画图 |
| E11 | early detection验证（如声称） | ☐ | 补充比例实验 |
| E12 | 可视化/案例分析 | ☐ | 补充图表 |
| E13 | 计算效率对比 | ☐ | 计时实验 |
| E14 | 实现可复现说明 | ☐ | 写README |
| — | 模块-挑战匹配度 ≥ B级 | ☐ | 重构模块 |
| — | 接口可行性无断裂 | ☐ | 修复维度/语义 |
| — | 消融可解释性 | ☐ | 补充因果命题 |

### 附录B：与调试手册的交叉引用索引

| 本手册章节 | 如果遇到问题 | 参见调试手册 |
|-----------|------------|------------|
| §5.3 基线复现 | 复现误差 > 5% | §3 DataAgent |
| §6.4 V1运行 | Loss不下降 | §4 ModelAgent / §5 TrainAgent |
| §6.6 V1交付 | 梯度不回传 | §4.2 M6 |
| §7.2 模块加入 | 维度不匹配 | §4.2 M2 |
| §7.2 模块加入 | 接口断裂 | §4.2 M10 |
| §8.2 消融设计 | 消融反直觉 | §6 AblateAgent |
| §8.5 统计检验 | t-test不会写 | §5.2 T9 |
| §11.2 智能体速查 | 各阶段症状匹配 | §2 问题→智能体快速匹配 |
| §11.3 五步法 | 检查清单执行 | §3-§6 各Agent检查清单 |
| §11.4 元调度器 | 多智能体冲突消解 | §7 元诊断调度器使用流程 |

### 附录C：各方向推荐基线速查

| 方向 | 首选基线 | 备选基线 | 数据集 |
|------|---------|---------|--------|
| 传播预测 | BuzzBloom | CasFlow, DeepHawkes | Memetracker, Weibo |
| 谣言检测 | KPG | RAGCL, GACL, EIN | Weibo16, Weibo22, Twitter |
| 传播建模 | CasFlow | NDM, Hawkesformer | Sina Weibo, APS |
| 社交机器人 | UnDBot | BotCGP | TwiBot-20, Midterm |

### 附录D：术语对照表（中英文）

| 中文术语 | 英文术语 | 首次出现章节 |
|---------|---------|------------|
| 信息瓶颈 | Information Bottleneck (IB) | §2.2 |
| 消融实验 | Ablation Study | §8.2 |
| 基线 | Baseline | §4.1 |
| 级联预测 | Cascade Prediction | §4.2 |
| 梯度消失 | Vanishing Gradient | §4.3 M6 |

---

> **记住**：实验是迭代过程，不是线性过程。遇到问题随时查阅《实验调试与自检手册》，按系统化的方法排查，不要凭直觉乱改。

---

*维护者：项目团队*  
*版本：v1.1*  
*配套文档*：《实验调试与自检手册》、《实验多智能体系统设计白皮书》
