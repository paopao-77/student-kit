# 数据集选择与准备计划

> 当前阶段：V0 环境搭建与数据准备  
> 已选主数据集：Weibo Rumor Dataset + Twitter15/16  
> 已加入补充完整数据集：PHEME  
> 已登记中文强候选数据集：AMiner Influence Locality  
> 目标：为 C1 传播动能辨识、C2 破圈预警、C3 闭环控制提供统一的数据输入。

## 1. 数据集选择结论

| 数据集 | 定位 | 为什么选它 | 支撑的实验 |
|---|---|---|---|
| Weibo Rumor Dataset | 中文主数据集 | 与中文谣言传播场景贴近，适合验证文本、传播树、时间戳和控制策略 | C1传播规模预测、C2破圈预警、C3控制仿真 |
| Twitter15/16 | 跨平台泛化数据集 | 国际常用谣言传播数据集，适合证明方法不是只对中文平台有效 | C1传播规模预测、C2破圈预警、C3控制泛化验证 |
| PHEME | 完整文本-时间-结构补充数据集 | 同时包含源帖文本、回复文本、真实时间戳、会话结构和真实性标注 | C1文本模块验证、C2真实时间预警、C3时滞控制验证 |
| AMiner Influence Locality | 中文微博大规模传播候选数据集 | 包含微博转发、用户关系、用户资料和原始推文内容，最贴近“异构社交生态”设定 | C1中文传播动力学、C2跨社区扩散、C3控制仿真；需先确认是否有谣言标签 |

## 2. 每个数据集必须整理的字段

| 字段类别 | 具体内容 | 用途 | 缺失时的处理 |
|---|---|---|---|
| 内容字段 | 原帖文本、转发/评论文本 | PLM文本编码，支撑C1 | 若缺少转发文本，至少保留原帖文本 |
| 用户字段 | 用户ID、粉丝数、关注数、认证状态、账号年龄等 | 用户属性编码，支撑异构社交生态建模 | 若缺失真实属性，用节点度、活跃度、传播位置替代 |
| 拓扑字段 | 传播树、转发边、评论边、社交关系边 | GAT/TGN动态图编码，支撑C1/C2 | 若缺少社交关系边，用传播树边构造近似拓扑 |
| 时间字段 | 原帖时间、转发时间、评论时间 | 时序传播日志、动态图快照、提前预警时间 | 必须保留；缺失严重则该样本剔除 |
| 标签字段 | 谣言/非谣言标签、事件ID | 监督训练与数据划分 | 标签不一致时统一映射为0/1或多类标签 |
| 破圈字段 | 社区ID、跨社区传播时间点、破圈事件标注 | C2破圈临界预警ground truth | 用Louvain/Leiden社区检测 + 跨社区边比例阈值构造 |

## 3. 统一数据格式

后续 loader 应尽量统一输出以下结构：

```text
sample_id
source_text
label
user_features
cascade_edges
social_edges
event_times
dynamic_snapshots
community_ids
breakout_time
```

其中：

| 输出项 | 说明 | 对应模块 |
|---|---|---|
| source_text | 原帖或事件文本 | PLM文本编码 |
| user_features | 用户属性或结构替代特征 | 用户编码/GAT |
| cascade_edges | 传播树或级联边 | GAT/TGN |
| event_times | 每次转发/评论的时间戳 | LSTM/TGN/SEIR |
| dynamic_snapshots | 按时间窗口构造的动态图序列 | TGN、破圈预警 |
| community_ids | 社区划分结果 | 跨社区拓扑变化检测 |
| breakout_time | 首次显著跨社区扩散时间 | C2预警标签 |

## 4. 数据预处理任务清单

| 编号 | 任务 | 产出物 | 状态 |
|---|---|---|---|
| D0-1 | 下载并解压 Weibo Rumor Dataset | `data/raw/BiGCN_Weibo/` | 已完成 |
| D0-2 | 下载并解压 Twitter15/16 | `data/raw/rumdetect2017/rumor_detection_acl2017/` | 已完成；源推文文本覆盖 100%，传播树已去重 |
| D0-3 | 统计样本数、用户数、边数、平均级联长度 | `data/processed/dataset_stats.csv` | 已完成 |
| D0-4 | 下载并解压 PHEME | `data/raw/PHEME/all-rnr-annotated-threads/` | 已完成 |
| D0-5 | 登记 AMiner Influence Locality 获取入口 | `data/raw/AMiner_InfluenceLocality/README.md` | 已完成，官方页面未暴露直接下载链接 |
| D0-6 | 统一标签编码 | `label_map.json` | 已完成 |
| D0-7 | 构建传播树/传播图 | `data/processed/{dataset}/edges.csv` | 已完成 |
| D0-8 | 构建时间窗口动态图快照 | `data/processed/{dataset}/dynamic_snapshots/snapshots.csv` | 已完成 |
| D0-9 | 做社区检测 | `data/processed/{dataset}/community_ids.csv` | 已完成，当前为传播分支启发式社区 |
| D0-10 | 构造破圈事件标签 | `data/processed/{dataset}/breakout_events.csv` | 已完成，当前为启发式初版 |
| D0-11 | 生成统一 train/val/test 划分 | `data/processed/splits/*.json` | 已完成 |
| D0-12 | 编写统一数据加载器 | `dataset_loader.py` | 已完成 |

## 5. 破圈事件标注初版规则

先采用可解释的启发式规则，后续再根据数据分布调阈值：

```text
若某级联在时间窗口 t 内满足：
1. 新增跨社区传播边数量明显上升；
2. 跨社区传播边占比超过阈值 theta_cross；
3. 涉及社区数从单一/少数社区扩展到多个社区；
则将 t 标注为 breakout_time。
```

建议初始阈值：

| 参数 | 初始值 | 后续搜索范围 |
|---|---:|---|
| 时间窗口大小 | 1小时 | 0.5小时、1小时、2小时、4小时 |
| theta_cross | 0.2 | 0.1、0.2、0.3 |
| 最小跨社区边数 | 5 | 3、5、10 |

## 6. 本周推进顺序

1. 下载两个数据集，并保留原始压缩包与README。
2. 写一个统计脚本，输出样本数、用户数、边数、级联长度、时间跨度。
3. 先不写复杂模型，只验证数据是否能被统一读取。
4. 完成时间切分，避免随机切分造成数据泄漏。
5. 为 Weibo 和 Twitter15/16 各抽取 3 个样本，手动检查传播树和时间戳是否合理。
