# 微博转发数据集字段说明

整理日期：2026-05-28  
数据目录：`E:\kuakedownload\weibodata`

## 0. 编码与读取说明

本数据集不是统一 UTF-8。读取时要特别注意：

| 文件/目录 | 推荐编码 | 说明 |
| --- | --- | --- |
| `root_content.txt` | GBK / GB18030 | 原始中文正文，UTF-8 读取会乱码 |
| `retweetWithContent\repost*.txt` | GBK / GB18030 | 带中文转发正文 |
| `retweetWithoutContent\total.txt` | GBK / GB18030 | 主要是数字和时间 |
| `weibocontents\*.txt` | GBK / GB18030 | readme、词表、哈希正文等 |
| `userProfile\user_profile*.txt` | GBK / GB18030 | 用户昵称、位置、简介为中文 |
| `diffusion\*.txt` | GBK / GB18030 | 主要是数字，按 GB18030 读取无问题 |
| `weibo_network\weibo_network.txt` | GBK / GB18030 | 主要是数字 |
| `graph_170w_1month.txt` | GBK / GB18030 | 主要是数字 |
| `topic-100\topic` | UTF-8 | 这个文件用 GBK 会出现 `缃戠珯` 这类乱码 |
| `topic-100\doc` | GBK / GB18030 或 UTF-8 均可 | 主要是数字和 ASCII |

PowerShell 读取中文样例：

```powershell
$enc = [Text.Encoding]::GetEncoding('GB18030')
$sr = [IO.StreamReader]::new('root_content.txt', $enc)
$sr.ReadLine()
$sr.Close()
```

## 1. ID 体系

这个数据集里同时存在两类 ID，分析时最容易混：

| ID 类型 | 示例 | 含义 | 映射方式 |
| --- | --- | --- | --- |
| 原始微博 ID / `mid` | `3515638699605834` | 微博平台原始微博 ID | 直接出现在内容文件中 |
| 原始用户 ID / `uid` | `1657151084` | 微博平台原始用户 ID | 用户画像里直接使用 |
| 内部微博 ID / `post_id` | `0`、`1000` | `diffusion` 传播日志内部编号 | `diffusion\repost_idlist.txt` 第 n 行对应 `post_id=n` |
| 内部用户 ID / `user_id` | `108051` | 传播日志、网络文件里的内部用户编号 | `uidlist.txt` 或 `diffusion\uidlist.txt` 第 n 行对应 `user_id=n` |

已确认规模：

| 项目 | 数量 |
| --- | ---: |
| 原微博数 | 300,000 |
| 内部用户数 | 1,787,443 |
| 主题模型文档数 | 299,795 |
| 用户画像记录数，估算 | 1,681,085 |
| 邻接网络边数，文件头给出 | 413,503,687 |

## 2. 文件清单

| 路径 | 内容 | 结构 |
| --- | --- | --- |
| `root_content.txt` | 原微博中文正文 | 2 行一条 |
| `weibocontents\Root_Content.txt` | 哈希后的原微博正文 | 2-4 行一条 |
| `weibocontents\Retweet_Content.txt` | 哈希后的转发正文 | 变长块 |
| `weibocontents\Weibo_Retweet_Num.txt` | 原微博转发数 | 1 行一条 |
| `weibocontents\WordTable.txt` | 词表 | 首行词表大小，后续 3 列 |
| `diffusion\repost_data.txt` | 转发时间序列 | 变长块 |
| `diffusion\repost_idlist.txt` | `post_id` 到 `mid` 映射 | 1 行一个 `mid` |
| `diffusion\uidlist.txt` | `user_id` 到 `uid` 映射 | 1 行一个 `uid` |
| `uidlist.txt` | `user_id` 到 `uid` 映射 | 与 `diffusion\uidlist.txt` 同规模 |
| `retweetWithoutContent\total.txt` | 不含正文的转发列表 | 2 行一条原微博 |
| `retweetWithContent\repost*.txt` | 含正文的转发分片 | 变长块 |
| `userProfile\user_profile1.txt` | 用户画像分片 1 | 14 字段逐行存储 |
| `userProfile\user_profile2.txt` | 用户画像分片 2 | 14 字段逐行存储 |
| `weibo_network\weibo_network.txt` | 用户关系网络邻接表 | 首行规模，后续邻接表 |
| `graph_170w_1month.txt` | 用户关系网络边表 | 3 列边表 |
| `topic-100\topic` | 100 个主题的关键词 | 3 列 |
| `topic-100\doc` | 每条微博的主题分布 | 变长列 |

## 3. 原微博中文正文

### `root_content.txt`

编码：GBK / GB18030  
规模：600,000 行，约 300,000 条微博

格式：

```text
mid
content
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `mid` | 整数/字符串 | 原始微博 ID |
| `content` | 字符串 | 原始中文微博正文 |

样例：

```text
3515638699605834
当前后左右都没有路时，命运一定是鼓励你向上飞了。
```

## 4. 哈希后的原微博正文

### `weibocontents\Root_Content.txt`

编码：GBK / GB18030  
规模：739,320 行，对应约 300,000 条原微博

格式：

```text
original_tweet_id
content_token_ids
[@ mention_list]
[link url]
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `original_tweet_id` | 整数/字符串 | 原始微博 ID，即 `mid` |
| `content_token_ids` | 空格分隔整数序列 | 微博正文中有意义词的哈希/词 ID 序列 |
| `@ mention_list` | 字符串，可缺失 | 原文中 @ 的用户名列表；无 @ 时可能没有该行，也可能只有 `@` |
| `link url` | 字符串，可缺失 | 原文中的链接 |

注意：

- 每条微博最少 2 行，最多 4 行。
- 正文不是自然语言，而是 token id 序列。
- token id 可结合 `weibocontents\WordTable.txt` 理解。

## 5. 词表

### `weibocontents\WordTable.txt`

编码：GBK / GB18030

格式：

```text
vocabulary_size
word_id    count    word
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `vocabulary_size` | 整数 | 词表大小，位于第一行 |
| `word_id` | 整数 | 词或哈希 token 的 ID |
| `count` | 整数 | 词频 |
| `word` | 字符串 | 中文词或符号 |

样例：

```text
2064652
14    14    二姑娘
20    1     贪军功
25    2     罗帅锅
```

## 6. 原微博转发数

### `weibocontents\Weibo_Retweet_Num.txt`

编码：GBK / GB18030  
规模：300,000 行

格式：

```text
original_tweet_id    number_of_retweets
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `original_tweet_id` | 整数/字符串 | 原微博 ID |
| `number_of_retweets` | 整数 | 该微博在数据集中记录到的转发数 |

样例：

```text
3464198275618249    190
3441315499418713    869
3484010254880722    34
```

## 7. 哈希后的转发内容

### `weibocontents\Retweet_Content.txt`

编码：GBK / GB18030  
结构：按原微博分块，每个块包含原微博头信息和若干条转发

格式：

```text
original_mid original_uid original_time retweet_num
N
retweet_uid retweet_time retweet_mid
retweet_content_token_ids
[@ mention_list]
[retweet user_list]
[link url]
...
```

原微博头字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `original_mid` | 整数/字符串 | 原微博 ID |
| `original_uid` | 整数/字符串 | 原微博作者 UID |
| `original_time` | 时间字符串 | 原微博发布时间，格式如 `2012-09-24-15:47:16` |
| `retweet_num` | 整数 | 原微博实际转发数 |
| `N` | 整数 | 当前文件中收录的转发条数 |

单条转发字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `retweet_uid` | 整数/字符串 | 转发用户 UID |
| `retweet_time` | 时间字符串 | 转发时间 |
| `retweet_mid` | 整数/字符串 | 转发微博 ID |
| `retweet_content_token_ids` | 空格分隔整数序列，可为空 | 转发正文的哈希 token 序列 |
| `@ mention_list` | 字符串，可缺失 | 转发正文中的 @ 列表 |
| `retweet user_list` | 字符串，可缺失 | 转发链中的用户列表；如果直接转发原微博，可能没有该行 |
| `link url` | 字符串，可缺失 | 转发内容中的链接 |

注意：

- 单条转发记录是 2 到 5 行。
- `retweet_content_token_ids` 即使没有有效词，也会保留一个空行。

## 8. 原始中文转发内容分片

### `retweetWithContent\repost*.txt`

编码：GBK / GB18030  
文件数：约 1,038 个 `repost*.txt` 分片

格式：

```text
original_mid    original_uid    original_time    retweet_num
N
retweet_uid    retweet_time    retweet_mid
retweet_content
...
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `original_mid` | 整数/字符串 | 原微博 ID |
| `original_uid` | 整数/字符串 | 原微博作者 UID |
| `original_time` | 时间字符串 | 原微博发布时间 |
| `retweet_num` | 整数 | 原微博总转发数或原始统计转发数 |
| `N` | 整数 | 该块中实际列出的转发条数 |
| `retweet_uid` | 整数/字符串 | 转发用户 UID |
| `retweet_time` | 时间字符串 | 转发时间 |
| `retweet_mid` | 整数/字符串 | 转发微博 ID |
| `retweet_content` | 字符串 | 原始中文转发正文；可能出现 `error` 或空内容 |

样例结构：

```text
3509517944222416    1857916052    2012-11-07-00:02:22    25975
5
1767461044    2012-11-07-12:08:50    3509700752315087
治愈一下！@z奥_vendredi //@木十喬Delling:无
```

注意：

- 这里的正文是原始中文/符号文本，不是哈希 token。
- 这些分片更适合做文本分析或转发链文本复原。

## 9. 转发传播日志

### `diffusion\repost_data.txt`

编码：GBK / GB18030  
规模：readme 说明包含 300,000 条微博；实际行数约 35,993,551 行

格式：

```text
post_id    number_of_reposts
timestamp  user_id
timestamp  user_id
...
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `post_id` | 整数 | 内部微博 ID，从 0 开始 |
| `number_of_reposts` | 整数 | 当前用户集合中唯一转发用户数 |
| `timestamp` | Unix 秒级时间戳 | 转发发生时间，从 `1970-01-01 00:00:00` 起算 |
| `user_id` | 整数 | 内部用户 ID，从 0 开始 |

样例：

```text
0    1798
1352244462    108051
1352249027    81604
```

重要说明：

- `number_of_reposts` 不是全微博网络总转发数，而是当前用户集合中的唯一转发用户数。
- 同一用户多次转发同一微博时，只保留第一次转发。
- `post_id` 用 `diffusion\repost_idlist.txt` 映射回原始 `mid`。
- `user_id` 用 `diffusion\uidlist.txt` 映射回原始 `uid`。

### `diffusion\repost_data_partition_1.txt`

`repost_data.txt` 的分区文件之一，格式同上。

### `diffusion\repost_data_partition_2.txt`

`repost_data.txt` 的分区文件之一，格式同上。

### `diffusion\repost_data_sample.txt`

样例文件，格式同 `repost_data.txt`。

## 10. 传播日志 ID 映射

### `diffusion\repost_idlist.txt`

编码：GBK / GB18030  
规模：300,000 行

格式：

```text
original_mid
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `original_mid` | 整数/字符串 | 原始微博 ID |

映射规则：

```text
第 n 行的 original_mid = diffusion\repost_data.txt 中 post_id 为 n 的原微博 ID
```

### `diffusion\uidlist.txt` 与 `uidlist.txt`

编码：GBK / GB18030  
规模：1,787,443 行

格式：

```text
original_uid
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `original_uid` | 整数/字符串 | 原始微博用户 UID |

映射规则：

```text
第 n 行的 original_uid = user_id 为 n 的原始 UID
```

## 11. 不带正文的转发列表

### `retweetWithoutContent\total.txt`

编码：GBK / GB18030  
规模：465,956 行，约 232,978 个原微博块

格式：两行一组。

```text
original_mid original_time original_uid retweet_num
retweet_uid retweet_time retweet_uid retweet_time ...
```

原微博头字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `original_mid` | 整数/字符串 | 原微博 ID |
| `original_time` | 时间字符串 | 原微博发布时间 |
| `original_uid` | 整数 | 原微博作者用户 ID。样例值落在内部用户 ID 范围内，建议按内部 `user_id` 理解 |
| `retweet_num` | 整数 | 原始统计转发数或该微博转发数 |

转发列表字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `retweet_uid` | 整数 | 转发用户 ID，样例值落在内部用户 ID 范围内 |
| `retweet_time` | 时间字符串 | 转发时间 |

注意：

- 第二行是若干组 `retweet_uid retweet_time`。
- 样例中第二行列出的转发对数通常小于第一行的 `retweet_num`，因此 `retweet_num` 更像原始统计总数，第二行是当前数据中采到的转发列表。

## 12. 用户画像

### `userProfile\user_profile1.txt` 与 `userProfile\user_profile2.txt`

编码：GBK / GB18030  
规模：两个分片合计约 1,681,085 条用户画像

文件开头的字段注释：

```text
# id
# bi_followers_count
# city
# verified
# followers_count
# location
# province
# friends_count
# name
# gender
# created_at
# verified_type
# statuses_count
# description
```

存储方式：

```text
id
bi_followers_count
city
verified
followers_count
location
province
friends_count
name
gender
created_at
verified_type
statuses_count
description

下一位用户...
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `id` | 整数/字符串 | 原始微博用户 UID |
| `bi_followers_count` | 整数 | 互相关注数 |
| `city` | 整数编码 | 城市编码 |
| `verified` | 布尔字符串 | 是否认证，`True` / `False` |
| `followers_count` | 整数 | 粉丝数 |
| `location` | 字符串 | 位置文本，如 `上海 长宁区` |
| `province` | 整数编码 | 省份编码 |
| `friends_count` | 整数 | 关注数 |
| `name` | 字符串 | 用户昵称 |
| `gender` | 字符串 | 性别，常见为 `m`、`f` |
| `created_at` | 时间字符串 | 账号创建时间 |
| `verified_type` | 整数编码 | 认证类型；`-1` 常见于非认证用户 |
| `statuses_count` | 整数 | 微博发布数量 |
| `description` | 字符串，可为空 | 用户简介 |

注意：

- 每个用户 14 个字段，逐行存储，用户之间通常有空行分隔。
- 用户画像记录数少于 `uidlist` 的 1,787,443，说明不是所有内部用户都有画像。

## 13. 用户关系网络：邻接表

### `weibo_network\weibo_network.txt`

编码：GBK / GB18030

首行：

```text
1787443    413503687
```

首行字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `num_users` | 整数 | 用户数 |
| `num_edges` | 整数 | 关系边数 |

后续每行格式：

```text
source_user_id    degree    target_user_id relation_flag target_user_id relation_flag ...
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `source_user_id` | 整数 | 源用户内部 ID |
| `degree` | 整数 | 后面列出的目标用户数量 |
| `target_user_id` | 整数 | 目标用户内部 ID |
| `relation_flag` | 0/1 | 关系标记，readme 未解释；样例中为二值 |

已校验：

- 第 2 行 `source_user_id=0`，`degree=296`，后续确实是 296 组 `target_user_id relation_flag`。
- 第 3 行 `source_user_id=1`，`degree=271`，后续确实是 271 组。

注意：

- `source_user_id` 与 `target_user_id` 都是内部用户 ID，需要通过 `uidlist.txt` 映射成原始 UID。
- `relation_flag` 的具体语义未在现有 readme 中说明，不能武断写成“是否互关”。

## 14. 用户关系网络：边表

### `graph_170w_1month.txt`

编码：GBK / GB18030

格式：

```text
source_user_id target_user_id relation_flag
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `source_user_id` | 整数 | 源用户内部 ID |
| `target_user_id` | 整数 | 目标用户内部 ID |
| `relation_flag` | 0/1 | 关系标记，语义未在 readme 中说明 |

样例：

```text
0 3 1
0 10 1
0 12 1
0 13 1
```

注意：

- 该文件与 `weibo_network\weibo_network.txt` 表达的是同类用户关系网络，但一个是边表，一个是邻接表。
- 如果做图算法，边表更直接；如果做按用户展开邻居，邻接表更省空间。

## 15. 主题模型

### `topic-100\topic`

编码：UTF-8  
规模：100 行

格式：

```text
topic_id    weight    top_words
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `topic_id` | 整数 | 主题编号，0-99 |
| `weight` | 浮点数 | readme 未解释，样例中为 `0.5` |
| `top_words` | 空格分隔词列表 | 该主题的代表词 |

样例，需按 UTF-8 读取：

```text
0    0.5    网站 用户 网络 服务 产品 信息 客户 广告 互联网 移动 营销 品牌 数据 技术 提供 平台 腾讯 百度 电子
1    0.5    后 医院 医生 时 死亡 手术 生命 严重 救 危险 治疗 不幸 患者 检查 发现 岁 男子 需 位
```

### `topic-100\doc`

编码：主要为数字和 ASCII；GBK/GB18030 可读  
规模：299,795 行

格式：

```text
doc_index    mid    topic_id    proportion    topic_id    proportion ...
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `doc_index` | 整数 | 文档序号 |
| `mid` | 整数/字符串 | 原微博 ID |
| `topic_id` | 整数 | 主题编号 |
| `proportion` | 浮点数 | 该主题在该微博中的占比 |

注意：

- readme 说明原始约 300,000 条微博中，去停用词后约 300 条被移除。
- 该文件每行是变长列，即一条微博后面跟多组 `topic_id proportion`。

## 16. 压缩包说明

当前目录中既有解压后的数据，也保留了压缩包：

| 压缩包 | 对应内容 |
| --- | --- |
| `weibocontents.rar` | `weibocontents` 目录 |
| `diffusion.rar` | `diffusion` 目录 |
| `userProfile.rar` | `userProfile` 目录 |
| `weibo_network.rar` | `weibo_network` 目录 |
| `graph_170w_1month.rar` | `graph_170w_1month.txt` |
| `root_content.rar` | `root_content.txt` |
| `topic-100.rar` | `topic-100` 目录 |
| `retweetWithoutContent.rar` | `retweetWithoutContent` 目录 |
| `retweetWithContentNew.7z.001`、`retweetWithContentNew.7z.002` | 带正文转发数据分卷压缩包 |

## 17. 常见使用路线

做传播时序：

1. 读 `diffusion\repost_data.txt` 得到 `post_id -> [(timestamp, user_id)]`。
2. 用 `diffusion\repost_idlist.txt` 把 `post_id` 转成原始 `mid`。
3. 用 `uidlist.txt` 把 `user_id` 转成原始 `uid`。

做用户网络：

1. 如果需要逐边处理，读 `graph_170w_1month.txt`。
2. 如果需要按用户展开邻居，读 `weibo_network\weibo_network.txt`。
3. 所有用户编号都是内部 `user_id`，需要用 `uidlist.txt` 映射。

做文本分析：

1. 原始中文原微博用 `root_content.txt`，编码 GB18030。
2. 原始中文转发文本用 `retweetWithContent\repost*.txt`，编码 GB18030。
3. 哈希后的正文用 `weibocontents\Root_Content.txt` 和 `weibocontents\Retweet_Content.txt`。
4. 词表用 `weibocontents\WordTable.txt`。

做主题分析：

1. 读 `topic-100\topic`，注意 UTF-8。
2. 读 `topic-100\doc` 获取每条微博的主题分布。
3. 用 `mid` 与 `root_content.txt` 或传播日志映射。

