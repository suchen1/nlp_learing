# NLP — 自然语言处理学习记录

跟随 **黑马程序员《AI 大模型 NLP 自然语言处理保姆级教程》** 的学习记录。

- 课程地址：<https://www.bilibili.com/video/BV14mdfBDE4Q/>
- 课程规模：共 **189 P**（课程导学 + Day01–Day12）
- 副标题：PyTorch 实现 Transformer 完整代码解析 + 预训练模型，一套搞定文本分类 / 翻译 / 情感分析等实战项目
- 内容路线：NLP 概念与文本预处理 → 词向量 → RNN/LSTM/GRU → 注意力机制 → Seq2Seq 英译法 → Transformer 逐层实现 → Fasttext 与预训练模型 → 迁移学习（BERT / ELMo / GPT）
- 分集清单与进度：见 [`课程进度.md`](课程进度.md)

本仓库是**个人学习记录**：代码由我本人跟着课程逐个练习编写并跑通，README 与课程进度表用来记录整体脉络和踩坑。
上一个项目 **《神经网络与深度学习》**（黑马，108 P，PyTorch → ANN → CNN → RNN）已学完，
项目地址：`D:\ML_Project\Deep Learn`。

## 学习进度

| 阶段 | 内容 | 分集 | 状态 |
| --- | --- | --- | --- |
| 导学 | 课程导学 | P1 | ✅ 已完成 |
| Day01 | 文本预处理：分词 / 词性标注 / NER / one-hot / CBOW | P2–P17 | ✅ 已完成 |
| Day02 | 词向量与文本分析 | P18–P35 | ✅ 已完成（词云 / 文本相似度两节无代码） |
| Day03 | 文本规范 + RNN / LSTM / GRU | P36–P54 | ✅ 已完成 |
| Day04 | 人名分类案例实战 | P55–P65 | ✅ 已完成（三模型对照：**GRU 88.2% > LSTM 87.3% > RNN 76.0%**） |
| Day05 | 注意力机制 | P66–P79 | ✅ 已完成（代码见 `注意力机制/`：位置打分版 + 内容比对版对照） |
| Day06 | Seq2Seq 英译法案例（上） | P80–P95 | ✅ **已完成**（P80–P92 视频 + P93–P95 **自己实现**的有 Attention 解码器；batch=1 五轮 92.5% → 改造成 **batch=32（pad+pack+mask）** 后 15 轮即到 **96.0%**、峰值 **98.5%**，同样时间可多训 7 倍轮次，见 `英译法/`） |
| Day07 | 英译法案例（下）+ Transformer 登场 | P96–P109 | ✅ **已完成**。P96–P104 的内容此前已**自己实现**（注意力解码器 / 训练 / 预测），P105 张量图 + P106 Transformer 由来为纯理论，P107–P109 Transformer 架构（编码器–解码器骨架） |
| Day08 | Transformer 逐层实现（上） | P110–P125 | ✅ **已完成**（`transformer架构/dm01`、`dm02`：词嵌入 / 位置编码 / 三角矩阵 / 掩码 / 注意力 / 多头 / FFN） |
| Day09 | Transformer 逐层实现（下） | P126–P143 | ✅ **已完成**（`dm02`~`dm09`：LayerNorm / 子层 / 编码器 / 解码器 / 输出 / 整体；完整模型 **45,677,544 参数 / 174.25 MB**，通过因果性·位置编码·padding·梯度四项行为测试） |
| Day10 | Fasttext 优化 + 预训练模型 + Transformers 库 | P144–P160 | ⬜ 未开始 |
| Day11 | Transformers 实战：pipeline 与自动模型 | P161–P172 | ⬜ 未开始 |
| Day12 | 迁移学习：BERT / ELMo / GPT | P173–P189 | ⬜ 未开始 |

## 知识点笔记

### Day01 文本预处理与词向量（P2–P17）

> 当前进度：Day01 视频已看完，下一步 **P18 = Day02-01 Word2Vec 之 SkipGram 词向量解释**。

#### 1. 分词（jieba）

| 模式 | 调用 | 特点 | 用在哪 |
| --- | --- | --- | --- |
| 精确模式 | `jieba.lcut(s)` | 最精确，没有冗余 | 文本分析、分类、情感分析 |
| 全模式 | `jieba.lcut(s, cut_all=True)` | 扫出所有能成词的组合，快，但有冗余和歧义 | 粗召回 |
| 搜索引擎模式 | `jieba.lcut_for_search(s)` | 在精确模式基础上把长词再切短 | 建倒排索引，提高召回 |

实测句 `我来到北京清华大学，顺便去了一趟北京市海淀区中关村`：

```
精确模式     ： 我/来到/北京/清华大学/，/顺便去/了/一趟/北京市/海淀区/中关村
全模式       ： 我/来到/北京/清华/清华大学/华大/大学/，/顺便/顺便去/了/一趟/北京/北京市/京市/海淀/海淀区/淀区/中关/中关村/关村
搜索引擎模式 ： 我/来到/北京/清华/华大/大学/清华大学/，/顺便/顺便去/了/一趟/北京/京市/北京市/海淀/淀区/海淀区/中关/关村/中关村
```

- `jieba.cut()` 返回**生成器**，`lcut()` 直接给列表
- **HMM 开关**：默认 `HMM=True`，用隐马尔可夫做**新词发现**，词典里没有的词也能拼出来
  - `他来到了网易杭研大厦` → `HMM=True`：网易/**杭研**/大厦；`HMM=False`：网易/**杭**/**研**/大厦
  - `李小福是创新办主任……` → `HMM=True`：**李小福**；`HMM=False`：李/小/福
- **自定义词典**：`jieba.load_userdict('userdict.txt')`，文件格式 `词语 词频 词性`
  - 动态增删：`add_word(word, freq, tag)` / `del_word(word)`
  - 词频微调：`suggest_freq(('中', '将'), True)` 让「中将」**裂开**；`suggest_freq('台中', True)` 让「台中」**合并**（必须在 `HMM=False` 下才看得出来，否则 HMM 的新词发现会盖掉词频的影响）
  - ⚠ 词典是**全局状态**，改完会影响后面所有输出，演示完记得改回来
- **繁体**：jieba 词典是简体词典，繁体能切但**不稳定**（实测「結巴分詞」→「結巴分 / 詞」），工业做法是先 `opencc` 繁转简（`t2s`）再分词

#### 2. 词性标注与 NER

- `jieba.posseg.cut(s)` 一步给出「词/词性」。常用标签：`n` 名词、`nr` 人名、`ns` 地名、`nt` 机构名、`nz` 其他专名、`v` 动词、`a` 形容词、`d` 副词、`m` 数词、`q` 量词、`t` 时间、`eng` 英文、`x` 标点
- jieba **没有真正的 NER**，只是拿 `nr/ns/nt` 这些词性标签近似，误识别很多，实测：
  - 「续航」被标成 `nr`（人名）、「人山人海」被标成 `ns`（地名）、「匡扶汉」被标成 `nr`
  - 该识别出来的人名却漏了（「刘备」没标出来）
- 想提高只能靠自定义词典 + 自定义词性；**真正的解法是 BERT + CRF 这类序列标注模型**（Day12 的方向）

#### 3. one-hot 的局限（为什么要学词向量）

- 维度 = 词表大小（几万～几十万），极其稀疏
- 任意两个不同的词内积都是 0：「猫」和「狗」的正交程度，跟「猫」和「宇宙」**一模一样** → 表达不出「相似」

#### 4. CBOW 原理（P17 重点）

- **方向别记反**：CBOW = 两边的上下文 → 预测中间词（完形填空）；Skip-gram = 中间词 → 预测两边
- 前向计算三步：
  1. 输入 `2×window` 个 one-hot（1×V），× W(V×N) —— one-hot 乘矩阵**等价于查表**，结果就是 W 的第 i 行
  2. CBOW 把这些上下文向量**平均**成一个 hidden（1×N）。注意是**平均不是拼接**，而且**这一层没有激活函数**（线性投影，不是 ANN 那种隐藏层）
  3. hidden × W'(N×V) → score(1×V) → softmax → p̂（全词表每个词分到一个概率）
- **标签和 loss 是两回事**：
  - 标签 = 中心词的 one-hot（等价说法：中心词的下标）
  - loss = 交叉熵 = **标量** `−log p̂[中心词]`，**不是** one-hot
  - 第一次跑 p̂[中心词] ≈ 1/V，loss ≈ ln V（V=10000 时约 9.2），训练就是把这个数往下压
- **反向传播到底改了谁**：
  - 输入是 one-hot → **W 只有被激活的那几行拿到梯度**（取平均时各除以 2×window），其余行纹丝不动
  - 输出是稠密的 softmax 分布 → **W' 几乎每一列都在动**（目标词那列往上推，其余词被压低）
  - 所以全词表 softmax 的开销主要压在 W' 上 → 原版用**负采样 / 层次 softmax** 替代（Day10 讲哈夫曼树与负采样）
- **两个权重矩阵**：

  | 矩阵 | 形状 | 含义 | gensim 里 |
  | --- | --- | --- | --- |
  | W | V×N | 第 i **行** = 词作为上下文（输入）时的向量 | `syn0` / `model.wv` |
  | W' | N×V | 第 j **列** = 词作为目标（输出）时的向量 | `syn1neg` |

  同一个词在这两个矩阵里是**两个不同的向量**；平时说的「词向量矩阵」指 W
- **词向量是副产品**：训练目标始终是「把目标词的概率顶上去」，向量只是被反复轻推塑造出来的中间产物
- 训练**不是**要让 hope 的向量等于 can 的向量，而是让**上下文用法相似的词**靠近（hope/wish、can/could）

#### 5. 样本数与 epoch（最容易想错的地方）

- 样本单位是**中心词的位置**，不是「词」。`1,2,3,4,5,6`、window=1 时：
  `(1,3)→2`、`(2,4)→3`、`(3,5)→4`、`(4,6)→5`，一共 **4 个样本**
- 边界有两种约定：
  - 严格版：中心词只在 `[1+window, N−window]` 里取，样本数 = `N − 2×window`
  - 边界收缩版：中心词取遍全句，窗口到句子边界自动变窄 → 样本数 ≈ 词数（原版 `word2vec.c` 和 gensim 用的是这个）
- Skip-gram 每个中心位置产生 `2×window` 个样本 → 同样 epoch 数下更新次数是 CBOW 的 `2×window` 倍（更慢，但小语料上效果常更好）
- 一个 epoch = 把**所有位置**过一遍。`epochs` 默认 **5**（gensim 与 `word2vec.c` 一致）；论文里 Google News（1000 亿词）只跑 **3 个 epoch**，学习率 0.025 线性衰减
- **语料越小轮数越多**（小语料 10–30 轮，大语料 3–5 轮）。衡量训练量请统一看「**总更新步数**」，比轮数实在
- 学习率**随进度线性衰减**（0.025 → 0.0001）：轮数给太少 = 还没退火完就停了
- 原版 C 实现的两个细节：**dynamic window**（每步的窗口在 `[1, window]` 里随机取，越近的词越常当上下文）、**subsampling**（高频词按概率直接丢掉）

### Day02 词向量与 WordEmbedding（P18–P35）

> 已全部完成（P18–P35）。下面记的是 P18–P25 这段的核心内容，P26–P34 的文本分析见 `文本分析/` 目录。

#### 1. FastText 实操（P19–P22）→ `文本张量表示/02_CBOW和skiggram.py`

- 环境：**`fasttext-wheel 0.9.2`**（官方 `fasttext` 在 Windows 上常要现场编译，`fasttext-wheel` 直接给轮子）
- 一句话训练：

  ```python
  model = fasttext.train_unsupervised(input='./data/ai20ag', model='cbow',
                                      epoch=10, lr=0.1, dim=100, ws=5, minCount=1)
  ```

  - `model='cbow'` / `'skipgram'` 就是两种词向量模型
  - `dim` 向量维度、`ws` 窗口、`epoch` 轮数、`lr` 学习率、`minCount` 最小词频
- 保存 / 加载：`model.save_model('./model/xxx.bin')` ｜ `fasttext.load_model('./model/xxx.bin')`
- 效果检验：`model.get_word_vector('money')`（拿到向量）、`model.get_nearest_neighbors('dog')`（找近邻词）
- ⚠ 两个 `.bin` 各 **972 MB**（dim=100 × 词表），属于**训练产物**，不要提交 git

#### 2. `nn.Embedding` 的本质：一张"可训练的查表"

- `nn.Embedding(num_embeddings=V, embedding_dim=N)` 内部就是一个 `(V, N)` 矩阵 `weight`
- 前向传入**整数下标**，等价于 `weight[idx]`；数学上等价于 `one-hot × W`，实现上**直接查行**（稀疏、高效）
- `embedding_dim=8` 是"每个词用 8 个数表示"（不是 8 个词），整表形状 `(V, 8)`

#### 3. 最容易出错的地方：词表编号 1-based ↔ Embedding 行号 0-based

| 对比项 | 起点 | 说明 |
| --- | --- | --- |
| `tokenizer.word_index` | **1** | 0 号不用。Keras 源码原话：`note that index 0 is reserved, never assigned to an existing word` |
| `nn.Embedding` 的行号 | **0** | 0 就是普通一行，不会自动变成 padding |

- 所以必须写 `num_embeddings = len(tokenizer.word_index) + 1`，那个 `+1` 就是**给 0 号腾预留位**；不写的话编号最大的词没有对应的行，一取就报 `IndexError: index out of range in self`
- 23 个词 → 矩阵是 **24 行**（"词表大小" = 24，不是 23）
- 看到的"词 1 显示在第 2 行"是**人眼从 1 数行、tensor 从 0 数行**造成的，不是错位
- 预留行在没用 padding 前**梯度恒为 0**（实测 `grad[0] = 0.0`，优化器走一步也不动），但它不是冗余：
  - `pad_sequences` 补齐时补的就是 **0**（实测 `[[1,2,3],[4,5]]` → `[[1,2,3],[4,5,0]]`）
  - `padding_idx=0` 会把它**初始化成全 0 并永久锁死梯度**（即使真的喂 0 进去也不更新）
- Keras 已经建好反向表 `tokenizer.index_word`（`{1: '学习', 2: '深度', ...}`），需要"编号 → 词"时直接用它

#### 4. TensorBoard 词向量可视化（P24）

```python
from torch.utils.tensorboard import SummaryWriter

my_summary = SummaryWriter(log_dir=r'D:\ML_Project\nlp\runs')   # ⚠ 必须是全英文路径
my_summary.add_embedding(embd.weight,
                         metadata=['<PAD>'] + list(tokenizer.word_index.keys()),
                         tag='word_embedding')
my_summary.close()
```

- `SummaryWriter` **不画图，只往磁盘写事件文件**；画图的是 TensorBoard —— 一个独立的本地网页应用（`localhost:6006`）
- `add_embedding(mat, metadata, tag)` 三个参数：
  - `mat`：`(N, D)` 的向量矩阵（这里是 `(24, 8)`，N 个点、每个 D 维）
  - `metadata`：**每个点一个标签**，第 i 个标签对应矩阵第 i 行
  - `tag`：这组嵌入在界面里的名字（可以写多组做对比）
  - TensorBoard 用 **PCA / t-SNE / UMAP** 把 D 维降到 3 维，画成可旋转的散点图，语义相近的词会聚成簇
- 落盘产物（实测）：`events.out.tfevents.*`、`projector_config.pbtxt`、`00000/word_embedding/metadata.tsv`、`tensors.tsv`
- ⚠ **metadata 个数必须等于行数**，否则 `AssertionError: #labels should equal with #data points`（24 行 vs 23 个词，所以要 `['<PAD>'] +`）
- ⚠ **metadata 的"顺序"靠不住**：`list(word_index.keys())` 恰好对，是因为 Keras 用 `dict(zip(排好序的词表, 1..N))` 构造、而 dict 保留插入顺序。一旦插入顺序和编号顺序分叉（重新排过、从 JSON 读回来），它会**静默贴错标签**（实测 24 个标签错 22 个，且一行错都不报）。稳妥写法是按编号回填：

  ```python
  metadata = [tokenizer.index_word.get(i, '<PAD>')
              for i in range(len(tokenizer.word_index) + 1)]
  ```

- 启动：`tensorboard --logdir=D:\ML_Project\nlp\runs` → 浏览器 `http://localhost:6006` → 顶栏切到 **PROJECTOR**
- ⚠ 现在展示的是**随机初始化**的词向量（还没训练），图上一团乱麻；要训练之后再看才有聚类意义

### Day03 文本规范 + RNN / LSTM / GRU（P36–P54）

> 已全部完成（P36–P54）。P50–P54 是"人名分类案例的数据准备"，和 Day04 是一个连贯的案例，笔记统一放在下面的 Day04。
>
> **Day03 的详细笔记单独成篇**（内容较长，不往 README 里塞）：
> - 📄 [`RNN模型/LSTM反向传播.md`](RNN模型/LSTM反向传播.md) — LSTM 到底有几组权重、四个门的梯度怎么更新、为什么 `c_t` 是"梯度高速公路"（全部带实测数字，文末附可复制的复现代码）
> - 📄 [`RNN模型/GRU门控与反向传播.md`](RNN模型/GRU门控与反向传播.md) — 重置门 `r` / 更新门 `z` 的公式与作用、PyTorch 的确切写法、为什么 `∂h/∂h` 是稠密矩阵、RNN·GRU·LSTM 四方梯度衰减实测

#### 1. RNN 的两个返回值（实测形状）

| 返回值 | 形状（`batch_first=True`） | 装的是什么 |
| --- | --- | --- |
| `output` | (句数, 句长, 隐藏维度) | **每个时间步**的隐藏状态，且**只含最顶层** |
| `hn` | (层数, 句数, 隐藏维度) | **每层最后一步**的隐藏状态 |

关键结论（均实测）：

- `output[:, -1, :]` 与 `hn[-1]` **逐位相同**（`torch.equal` 为 True）——`hn` 就是 `output` 的最后一行，不是两份计算。
- `output` 的第 t 行 = "只喂前 t+1 个词时的 `hn`"（把序列截断单独跑一遍即可验证）。
- **分类**取最后一步（`hn[-1]` 或 `output[:, -1, :]`）；**词性标注 / 预测下一个词**这类每个位置都要输出的任务才取全部。
- `output` **不跨批次累积**：第二次调用只返回这一批的词，想拼接得自己 `torch.cat`；能带过批次的只有 `hn`（要手动当 `h0` 传回去）。
- `hn` 是**压缩记忆**：改第 1 个词只让 `hn` 动 0.172，改第 4 个词动 0.987；序列拉到 60 个词时，第 1 个词的影响在 float32 下已下溢为 0。

#### 2. 权重共享 ≠ 冻结

| 说法 | 含义 |
| --- | --- |
| 权值共享 | 一句话里**每个词都翻同一本手册**：T=1 和 T=60 的参数量都是 55 个，不会变成 55×T |
| 更新时机 | 整批读完、`backward()` 之后再 `step()` **改一次**；前向过程中权重一个字都不变 |

一句话：**"共享"是每个词用同一套参数；"更新"是一个 batch 改一次。** 门再多、句子再长，也是 `optimizer.step()` 一次全更新。

#### 3. 梯度在时间上是累加的

同一套 `W_ih` 在每个时间步都收到一份梯度，`backward()` 把它们**加到同一个 `.grad` 上**（分 5 次反传再相加 == 一次反传，实测一致）。注意：梯度**张量**可以相加，但各自**模长不能直接相加**（方向不同会抵消）。

#### 4. LSTM 的反向传播

详见 [`RNN模型/LSTM反向传播.md`](RNN模型/LSTM反向传播.md)。一句话版：

> LSTM 有 **4 组门**（`i` / `f` / `g` / `o`，PyTorch 把它们拼成一个 4H 的矩阵存）。反向传播走两条路：`h` 那条和 RNN 一样；**`c` 那条只乘遗忘门 `f`（对角标量）**，不乘权重矩阵、不过 tanh 导数——这就是它比 RNN 更能记住长距离的原因。实测每步衰减：RNN ×0.530、LSTM ×0.644、LSTM（遗忘门偏置=1）×0.839。

#### 5. GRU 与 LSTM 的区别（P48–P49）

`nn.RNN / nn.GRU / nn.LSTM` 三者实测对比（都取 `input_size=5, hidden_size=10`）：

| 模型 | 门的组数 | `weight_ih_l0` | `weight_hh_l0` | 总参数量 | 返回值 |
| --- | --- | --- | --- | --- | --- |
| `nn.RNN` | 1 组 | (10, 5) | (10, 10) | 170 | `(output, hn)` |
| `nn.GRU` | **3 组**：重置门 r、更新门 z、候选状态 n | (30, 5) | (30, 10) | **510** | `(output, hn)` |
| `nn.LSTM` | **4 组**：i / f / g / o | (40, 5) | (40, 10) | **680** | `(output, (hn, cn))` |

参数量正好是 **1 : 3 : 4**（170 / 510 / 680）——这就是"门"直接换算成参数量的方式。

**两个本质区别**：

1. **GRU 没有独立的记忆细胞 `c`**，只有 `h`。所以它返回 `(output, hn)`，**不像 LSTM 返回 `(output, (hn, cn))`**——写代码时这层嵌套元组最容易漏（实测返回类型：GRU 是 `Tensor, Tensor`；LSTM 是 `Tensor, tuple`）。
2. **GRU 用两个门干三件事**：更新门 `z` 同时管"忘旧"和"记新"（`h_t = (1-z_t)⊙h_{t-1} + z_t⊙n_t`），重置门 `r` 控制上一步的 `h` 有多少参与候选计算；而 LSTM 的遗忘门和输入门是两个独立的门。

**怎么选**：GRU 参数少 1/4、训练更快，短序列上常常和 LSTM 打平；LSTM 的记忆细胞通路更"直"，长序列（几百步以上）通常更稳。

> **两个门的细节与反向传播**见 [`RNN模型/GRU门控与反向传播.md`](RNN模型/GRU门控与反向传播.md)。三条最实用的结论先放这儿：
> 1. **PyTorch 的写法是 `h_t = (1-z_t)⊙n_t + z_t⊙h_{t-1}`**（`z` = 保留旧记忆的比例），教材上另一种写法 `z` 的含义正好相反，抄混了前向就对不上。
> 2. **重置门 `r` 只作用于 `W_hn·h_{t-1} + b_hn` 这一项**，不是作用在整个候选状态上。
> 3. GRU 的 `∂h_t/∂h_{t-1}` 是**稠密矩阵**（因为 `h` 既是记忆又参与门的计算），不像 LSTM 的 `∂c/∂c` 是严格对角阵——实测每步梯度衰减 **GRU ×0.613 < LSTM ×0.663**，长距离记忆 LSTM 略胜。

### Day04 人名分类案例实战（P55–P65）

> 一个完整闭环：全局字母表 → 读数据 → Dataset → DataLoader → RNN/LSTM/GRU 三套模型 → 训练 → 日志 → 对比图 → 预测。
> 代码在 [`RNN模型/04_人名国家识别.py`](RNN模型/04_人名国家识别.py)，实验日志与图在 [`RNN模型/log/`](RNN模型/log)。

#### 1. 数据与形状链路（这张表值一章）

| 环节 | 形状 | 说明 |
| --- | --- | --- |
| 原始一行 | `'Abl\tCzech'` | 名字 + 国家，20,074 行 / 18 个国家 |
| `dataset[i]` | `(名字长度, 57)` | 每字符 one-hot，57 = `string.ascii_letters + " .,;'"` |
| `DataLoader(batch_size=1)` | `(1, 名字长度, 57)` | 默认 collate 自动加 batch 维 |
| `nn.RNN(batch_first=True)` 的 `out` | `(1, 名字长度, 128)` | 每个时间步的隐藏状态（只含最顶层） |
| `hn` / `out[:, -1, :]` | `(1, 128)` | 最后一步 = 整条名字的"记忆" |
| `self.fc(...)` | `(1, 18)` | 18 个国家的 logits |

**所以本项目不需要 `unsqueeze`**——DataLoader 已经把 batch 维补好了。课程里那句 `input.unsqueeze(1)` 是因为它自己造的输入是 2 维 `(6, 57)`（它的 `nn.RNN` 也没写 `batch_first=True`）。**在 DataLoader 之后再 `unsqueeze(1)` 会变 4 维直接报错**：`ValueError: RNN: Expected input to be 2D or 3D, got 4D tensor instead`。

#### 2. `hn` 和 `out[:, -1, :]` 是不是一回事

**单向时是**（实测 `torch.equal` 逐位相同），不管几层。差别只在健壮性：

| 配置 | `out[:, -1, :]` | `hn[-1]` | `hn[0]` |
| --- | --- | --- | --- |
| 单层单向（本项目） | ✅ | ✅ | ✅ |
| 多层单向 | ✅ | ✅ | ❌ 静默取浅层 |
| 双向 `bidirectional=True` | ⚠️ 反向那半只见过最后一个字 | ❌ 只有反向那一半 | ❌ |
| 忘了 `batch_first` | ❌ 静默错形状 | ✅ | ✅ |

**结论：写 `self.fc(hn[-1])`**——层数和 `batch_first` 双免疫。双向分类才需要 `torch.cat([hn[-2], hn[-1]], dim=-1)`。

#### 3. 什么时候需要返回 `hn`

> **喂进去的是"完整的一条数据"，还是"某条数据的一小段"？完整 → 不用 hn；一小段 → 必须 hn。**

| 场景 | 要 hn 吗 | 谁在循环 |
| --- | --- | --- |
| 分类 / 回归（本项目） | ❌ | `nn.RNN` 内部自己循环完 |
| 同一条序列拆多次喂（逐字符、长文本切片） | ✅ | 循环写在你手里 |
| 文本生成（逐词吐字） | ✅ | 同上 |

⚠️ **hn 绝不能跨"不同样本"携带**——实测：带着上一条人名的 `hn` 去读下一条，数学上等价于**把两条无关人名当成一句话读**（`torch.equal` 验证过）。课程 P57 的测试函数演示了"一次性送"和"一个字符一个字符送"两种写法，两者结果相同（差 ~1e-7 的 float 误差）。

#### 4. LSTM 的初始状态：`(None, None)` 会崩

```python
self.rnn(x, (None, None))   # ✗ AttributeError: 'NoneType' object has no attribute 'dim'
self.rnn(x)                 # ✓ 不传 = 自动全 0
self.rnn(x, None)           # ✓ 等价于不传
self.rnn(x, (h0, c0))       # ✓ 显式全 0 与"不传"结果完全相同
```

PyTorch 只特判 `hx=None` **本身**，组装成 `(None, None)` 它就不认了。分类任务推荐写成 `def forward(self, x, hx=None)`，三个模型就能用同一句调用。

#### 5. LogSoftmax / NLLLoss / CrossEntropyLoss

| 写法 | 数值 | 判定 |
| --- | --- | --- |
| `CrossEntropyLoss(logits)` | 2.657051 | ✅ 本项目用的 |
| `NLLLoss(LogSoftmax(logits))` | 2.657051 | ✅ 与上面**完全等价** |
| `CrossEntropyLoss(LogSoftmax(logits))` | 2.657051 | ⚠️ 多余但结果一样（log_softmax 幂等） |
| `CrossEntropyLoss(softmax(logits))` | 2.876873 | ⚠️ 数值不同，多套了一层 |
| `NLLLoss(softmax(logits))` | **-0.070155** | ❌ 错，损失变负数 |

规则：**CrossEntropyLoss 吃原始 logits；NLLLoss 必须配 LogSoftmax。** 所以 `self.softmax = nn.LogSoftmax(dim=1)` 保持注释掉是对的。

#### 6. 为什么 batch_size 只能是 1（以及 GPU 为什么帮不上忙）

```
各人名长度: Abl=3  Adsit=5  Ajdrna=6  Alt=3  Antonowitsch=12  Antonowitz=10    （2~19，共 18 种）

batch_size=2 -> RuntimeError: stack expects each tensor to be equal size,
                but got [3, 57] at entry 0 and [5, 57] at entry 1
```

变长序列不能直接 stack，所以这个案例只能 b=1。**代价是 GPU 完全用不上**（实测 RTX 5060 Ti）：

| batch_size | CPU | GPU | 结论 |
| --- | --- | --- | --- |
| **1** | 0.798 ms/step | 0.818 ms/step | **GPU 反而略慢（0.98×）** |
| 32 | 1.960 ms/step | 0.860 ms/step | GPU 2.28× |
| 256 | 3.500 ms/step | 0.900 ms/step | GPU 3.89× |

想真正用上 GPU，得先做 padding + `pack_padded_sequence`（或按长度分桶）把 batch 提到 32 以上。

#### 7. 三模型对照实验结果（P60–P63）

20,074 条 / 18 个国家，5 轮，Adam lr=0.001，batch_size=1：

| 模型 | 耗时 | 最终损失 | 第1轮 | 第2轮 | 第3轮 | 第4轮 | 第5轮 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **GRU** | 142.3s | **0.5650** | 71.8% | 80.9% | 84.1% | 86.4% | **88.2%** |
| **LSTM** | 148.8s | 0.6116 | 69.5% | 79.8% | 83.0% | 85.5% | 87.3% |
| **RNN** | 140.5s | 0.8979 | 65.6% | 72.0% | 74.6% | 75.6% | 76.0% |
| （基线：全猜 Russian） | — | — | — | — | — | — | 46.9% |

**结果和理论对得上**：GRU 只有 3 组门、LSTM 有 4 组（参数量 510 : 680），小数据上 GRU 常常打平或略胜——这次是**略胜且更快**；RNN 低 12 个点、损失高 0.33，正是"梯度衰减最快"的直接体现（实测每步衰减 **RNN ×0.513 < GRU ×0.613 < LSTM ×0.663**，见 Day03 笔记）。

⚠️ **局限（诚实记录）**：这三个数字是**训练集准确率**——`my_Dataloader` 用了整个文件、没有划分训练/验证集，所以还不能说明泛化能力。下一步该切出 10~20% 验证集（人名分类这种"记忆型"任务尤其容易虚高）。

#### 8. 日志与可视化的做法

训练时每个模型写一份 JSON，画图只读 JSON：

```
RNN模型/log/
├── rnn.json / lstm.json / gru.json   每个含：耗时、loss_list、acc_list、per_epoch_*、超参数
├── loss_compare.png                  损失下降对比
├── acc_compare.png                   准确率对比（含 46.9% 基线）
├── acc_per_epoch.png                 每轮准确率柱状对比
└── time_compare.png                  耗时对比
```

JSON 字段约定（这样 `plot_train_logs` 之类的代码读 `loss_list` 就永远是对的）：

| 字段 | 含义 |
| --- | --- |
| `loss_list` / `acc_list` | **修正后**的全局平滑曲线（每 100 个样本一个点），画图直接用这两个 |
| `per_epoch_loss` / `per_epoch_acc` | 每轮曲线（每轮 200 个点，轮内重新起算） |
| `elapsed_seconds` / `epochs` / `lr` / `batch_size` | 超参数与耗时 |
| `raw_*_buggy` | 当初写错口径时的原始输出，**只留档，不要用**（原因见踩坑记录第 9 条） |

**图和训练解耦**——重画不用重训；日志本身就是"实验可复现"的证据（三个 JSON 共约 263 KB，已提交）。

### Day05 注意力机制（P66–P79）

代码：[`注意力机制/01_代码实现.py`](注意力机制/01_代码实现.py) ｜ 演示动画：[`注意力机制/注意力步骤动画.html`](注意力机制/注意力步骤动画.html)

#### 1. Q / K / V 到底是什么

- **同一个输入 × 三个不同的矩阵**：`q = x·W_Q`、`k = x·W_K`、`v = x·W_V`。**原始词向量不是其中任何一个**。
- 名字来自"字典查找"的类比：硬查找只返回**一个** value，注意力是**加权混合所有** value。
- 实测三者的形状（Seq2Seq 解码一步）：

  | | 来源 | 形状 |
  | --- | --- | --- |
  | Q | 解码器当前状态 `(1,1,128)` | `(1,1,64)` ← 只解 1 步 |
  | K | 编码器输出 `(1,32,128)` | `(1,32,64)` ← 32 个词 |
  | V | 编码器输出 `(1,32,128)` | `(1,32,64)` ← 同形状但参数不同 |

- **`k [1,32,64]` 怎么来的**：`32 = 源句词数`（序列长度），`64 = 投影维度`（`W_K` 的输出大小，自己定）。
  路径 A（Seq2Seq）`词id → embedding(1,32,128) → 编码器 GRU → W_K → (1,32,64)`；
  路径 B（Transformer）`embedding → W_K → (1,32,64)`。实测 `W_k(Henc[0,5]) == K[0,5]` ✓

#### 2. 注意力四步 + 两种算分数的算法

```
① 算分数 → ② softmax → ③ 加权求和 V → ④ 与 Q 融合
```

只有第 ① 步有两种流派，后面三步完全一样：

| 算法 | 公式 | 谁在用 |
| --- | --- | --- |
| **加性**（Bahdanau） | `v · tanh(W[q ; k])` | Bahdanau 原论文、小任务；**课程的 `Linear(cat(q,k))` 是它的极简版** |
| **点积**（Luong） | `q · k / √d` | Transformer 及所有现代模型 |

- **`bmm(q, k^T)` 不神秘——矩阵乘法 = 批量点积**。手算验证：`q=[1,2]`，三个 k 为 `[3,4]/[5,6]/[7,8]` → 点积 `11/17/23`，`bmm` 一次全算出 ✓
- `k.transpose(-2,-1)` 的作用：把"一个词一行"变成"**一个词一列**"，矩阵乘法才能按词取。
- `bmm` 要求两个参数都是 3D → 必须 `unsqueeze`；`matmul`/`@` 在 batch=1 时能用但 batch>1 会静默给出 `(B,B,d)`；
  `einsum('bw,bwd->bd', w, V)` 不需要 unsqueeze 且完全并行（CPU 实测 3.429 ms vs bmm 3.355 ms vs 逐元素乘加 16.026 ms）。

#### 3. ★ 课程的 attention 是"位置打分"，不是"内容比对"

课程/tutorial 的写法是 `self.atten = nn.Linear(query_dim + key_dim, word_count)`，**输入是 `[q ; k]` 两个向量，输出 `word_count` 个分数**——第 j 个分数由第 j 行权重算出：

```
score_j = w_j · [q ; k] + b_j      ← w_j 是训练完固定下来的常数
```

- **每个位置一套参数**，且 32 个分数只由 `q` 和**那一个共享的 k** 决定
- **完全不用到 v 的内容**：实测把 V 整个换掉，注意力权重**逐位完全相同**
- 实测：改那个唯一的 k → **32 个权重全变**（因为这个 k 参与了每一行）

**真正的内容比对要 K 是每个词一个**（`(1,32,64)`）：

```python
scores = torch.bmm(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)   # (1,1,64)@(1,64,32) → (1,1,32)
attention_weights = F.softmax(scores, dim=-1)
attention_output  = torch.bmm(attention_weights, v)
```

实测：**只改第 5 个词的 k → 变化的分数下标恰好是 `[5]`** ✓ 这才叫"拿 q 去和每个词的 k 比相似度"。
两版代码都在 [`注意力机制/01_代码实现.py`](注意力机制/01_代码实现.py) 里，跑一遍 `__main__` 能直接看到差别。

| | 位置打分（课程） | 内容比对（标准） |
| --- | --- | --- |
| 分数输入 | `[q ; 一个共享的 k]` | `[q ; k_j]` |
| 改一个词的 k | 32 个分数全变 | 只有那一个变 |
| 语序变化 / 变长句子 | ❌ 失效 | ✅ 天然支持 |
| 谁在用 | 官方 tutorial、本课程 | Transformer、所有现代模型 |

#### 4. 除以 `√d` 是**缩放**，不是装饰

`q·k` 是 d 个乘积之和，若分量均值 0 方差 1，则**点积的方差 = d、标准差 = √d**（d=64 实测：方差 64.76、标准差 8.05，理论 √64=8.00）。

数值一大，softmax 就饱和 → 梯度消失：

| d | 不缩放：最大权重 | 熵 | **梯度模长** |
| --- | --- | --- | --- |
| 16 | 0.62 | 1.07 | 0.459 |
| 64 | 0.89 | 0.47 | 0.136 |
| **256** | **1.0000** | **0.000** | **0.00000** ← 梯度精确为 0，训练直接卡死 |
| 1024 | 0.9997 | 0.003 | 0.00041 |

**除以 √d 之后**：不管 d 是 16 还是 1024，最大权重稳定在 0.13~0.25、熵 ~2.9、梯度 0.77~0.89 ✓
→ 这就是 Transformer 论文里 **"Scaled** Dot-Product Attention" 那个 Scaled。

#### 5. 自注意力 vs 交叉注意力：**公式一模一样，只是喂什么不同**

```python
def attention(q, k, v):                       # 同一段代码
    scores = bmm(q, k.T) / √d;  w = softmax(scores);  return bmm(w, v)

attention(X,     X,       X      )   # 自注意力：三个都传同一句话 → 分数是【方阵】(32,32)
attention(dec_h, enc_out, enc_out)   # 交叉注意力：q 传解码器  → 分数是【长条】(1,32)
```

**判断方法：看 Q、K、V 从哪来。** 方阵 = 自注意力（32 个词两两相比），长条 = 交叉注意力。
把 `q` 也换成 `(1,32,64)` 且 `k = v = q` → **代码一字不改就变成自注意力**。

#### 6. 无注意力解码器：中间语义只注入一次

- **中间语义 = 编码器最后一个时间步的隐藏状态**，形状 `(1, 1, H)` = `[层数, batch, 隐藏维度]`
- 实测 `enc_out[0,-1,:]` 和 `enc_hidden[-1]` **完全相同**（单向 GRU）
- 它是 `decoder(tmp, hidden)` 的**第二个参数**。⚠️ **不能写 `hidden[0]`**——那会变成 2 维，实测报
  `For batched 3-D input, hx should also be 3-D but got 2-D tensor`

```
第 0 步：输入 SOS，hidden = 编码器的 hidden   ← 中间语义【只在这里注入一次】
第 1 步起：输入上一个词，hidden = 解码器自己的
```

**第 1 步之后编码器就再也用不到了** —— 这就是**信息瓶颈**，源句越长信息丢得越多。
注意力机制正是为此而生：每一步都重新对 `enc_out` **全部**算权重。带注意力的解码器 `input_size = embd + hidden = 256 + 512 = 768` 就是这个"拼接上下文"的体现。

#### 7. 训练循环的两件必修课（P92 阶段踩到的）

| 事情 | 结论 |
| --- | --- |
| **SOS 与右移** | 解码器输入必须是 `[SOS] + y[:-1]`，目标 `y`。**不塞 SOS 的硬证据**：实测 SOS 那一行的 embedding 梯度**全为 0**，永远停在随机初始化上 → 推理时第一步喂 SOS 直接崩（详见踩坑记录第 11 条） |
| **decoder 里不能加 softmax** | `CrossEntropyLoss` 自带 `log_softmax`。加了之后实测 loss 偏大、**梯度只剩 16%**（0.0598 vs 0.3742）→ 训练明显变慢且难查（详见踩坑记录第 10 条） |

**Teacher Forcing 与 Free-running 是两件正交的事：**

| | 加 SOS？ | 每步喂什么 | 用在哪 |
| --- | --- | --- | --- |
| **正确训练** | ✅ | **真值** `y[0][j]` | Teacher Forcing |
| **正确推理** | ✅ | **自己的预测** `topi` | Free-running |

课程代码喂真值（对）但**没加 SOS**（缺）；自己最初写的那版加了 SOS（对）但喂的是自己的预测（训练时不该这么干）。
**两件都要对才是正确训练。** 实测 free-running 一旦某步猜错，下一步输入就是错的 → 错误累积（玩具模型上直接退化成 `0↔6` 死循环）。

**训练时循环其实不是必须的**：一次喂整句 `decoder(dec_input, hidden)` 与逐步循环**结果完全一致**（实测最大差异 `6.6e-7`），因为 GRU 本来就是从左到右处理。循环留给推理用。

### Day06 Seq2Seq 英译法案例（P80–P95）

代码：[`英译法/translate.py`](英译法/translate.py) ｜ 数据：[`英译法/data/eng-fra-v2.txt`](英译法/data/eng-fra-v2.txt)

#### 1. 数据与词表

| 项 | 值 |
| --- | --- |
| 句对 | **63,594** 行（英文 ↔ 法文，tab 分隔） |
| 英文词表 | **2803**（含 SOS=0 / EOS=1，真实词从 2 开始） |
| 法文词表 | **4345** |
| 句长 | 英文 3~9 词、法文 2~9 词，均值 6 词 → `MAX_LENGTH=10` 刚好够 |

**验证过**：所有行都有 tab、规范化后无空句、`split(' ')` 不产生空串；`nn.Embedding(count, dim)` 的合法编号恰好 `0 ~ count-1` ✓

#### 2. 模型结构（自己实现的有注意力解码器）

```python
EncoderRNN:  Embedding(2803, 256) → GRU(256, 512)
Decoder_Attention:
    Embedding(4345, 256) → Dropout(0.1)
    score = bmm(hidden^T, encoder_outputs^T) / √512      # ★ Q·K，不是课程的位置打分
    attn_weights = softmax(score, -1)
    context = bmm(attn_weights, encoder_outputs)
    combined = cat(embedded, context) → attn_combine(768→512) → relu
    GRU(512, 512) → Linear(512, 4345)
```

**关键设计**：用 `attn_combine` 把 768 维压回 512，所以 **GRU 的 `input_size` 保持 512** 不用变宽（比官方新版教程的做法更干净）。

#### 3. ★ 实验结果：正确率随轮次的变化

**评测方法**：200 句固定测试集（seed 固定，各轮共用），**匹配任一参考译法即算正确**（其中 70 句是"一对多"）。

| 轮次 | 正确率 | 相比上一轮 |
| --- | --- | --- |
| 第 1 轮 | 33.0% | — |
| 第 2 轮 | 59.5% | **+26.5** |
| 第 3 轮 | 72.5% | **+13.0** |
| 第 4 轮 | 82.5% | **+10.0** |
| **第 5 轮** | **92.5%** | **+10.0** |

**33% → 92.5%，5 轮提升 2.8 倍**；提升幅度递减（+26.5 → +13 → +10 → +10）。

> **训练量说明**：`batch=1` 下每轮约 11 分钟（10.7 ms/步 × 63,594 步），跑完 5 轮约 **57 分钟**，故到此为止 —— `batch=1` 是这里最大的性能瓶颈（上 batch 能快 7~12 倍，代价是要补 pad/pack/mask）。

**观察到的现象：**

- 大部分句子在**第 2 轮**就学会了（第 1 轮只学到最基础的对应关系）
- **罕见词的长尾问题会随训练轮次自愈**：`i am from shizuoka .` 前 4 轮都输出 `je suis de suis .`（罕见词 `shizuoka` 退化成高频词 `suis`），**第 5 轮终于输出正确的 `je suis de shizuoka .`**
- `we re in love .` → `nous sommes en .`（不是任何参考答案）—— 因为这句在训练集里**有 3 个不同的法语答案**，模型学成了"混合体"。这类"一对多"用多参考匹配也救不回来，是数据固有的不确定性

#### 4. ★ 超参数：`lr` 是最关键的一项

| 配置 | 1000 步后的表现 |
| --- | --- |
| **Adam lr=0.01** | **发散**（loss 8.87 → 22.22，越训越大） |
| Adam lr=0.001 | 正常（8.4 → 3.06） |
| **Adam lr=0.0001（最终采用）** | 稳但慢，需要更多轮次补偿 |

**训练 10 轮的实际配置**：`lr=0.0001`、`batch_size=1`、`teacher_forcing_ratio=0.5`、`epoch=10`，约 **10.7 ms/步** → 完整 10 轮约 **1.9 小时**。

详见踩坑记录第 12 条。

#### 5. 训练循环的三个易错点（都已实测）

| 事情 | 结论 |
| --- | --- |
| **SOS 与右移** | 输入 `[SOS] + y[:-1]`，目标 `y`。不塞 SOS 时 loss 反而更低（抄输入更容易），但推理崩 —— 见踩坑记录第 11 条 |
| **decoder 里不能加 softmax** | `CrossEntropyLoss` 自带 `log_softmax`，加了梯度只剩 16% —— 见踩坑记录第 10 条 |
| **推理时的 batch 维** | `evaluate` 里 `input_tensor` 必须 `.unsqueeze(0)`，否则 `embedding(1维)` → 2 维 → GRU 报 `For unbatched 2-D input, hx should also be 2-D but got 3-D` |

#### 6. ★ batch 化改造：PAD + pack + mask（batch=1 → 32，每句快 20 倍）

> **动机**：`batch=1` 时每轮 9.8 分钟，跑 5 轮就快一小时。改成 `batch=32` 后每轮 **0.7 分钟**。

##### 6.1 四个问题、四个解法（**不是四选一，各修一处**）

| # | 症状 | 解法 | 作用在哪一段 |
| --- | --- | --- | --- |
| ① | `stack expects each tensor to be equal size` → 组不了 batch | `my_collate` **补齐** | **DataLoader** |
| ② | GRU 把 PAD 也读了 → `hidden` 被污染 | `pack_padded_sequence` | **编码器的 GRU** |
| ③ | 注意力把权重分给 PAD | `masked_fill(-inf)` | **解码器的 softmax** |
| ④ | 模型花力气学"预测 PAD" | `ignore_index=PAD` | **损失函数** |

**★ 为什么 ② 做了还要 ③**（最容易晕的一点）：

```
pack            → GRU 跳过 PAD，hidden 干净了 ✓
pad_packed_sequence → 但要把输出变回 (B, L, H) 才做得了 bmm，PAD 位置填 0
                      ↓
                 这些 0 仍是"候选位置"！softmax 只看"有几个候选"，不管值是 0
                      ↓
                 所以还要 mask 把它们从分母里去掉 ✓
```

实测：`pack` 后 `.data` 是 `(11, 512)`（只有真实词），但 `pad_packed_sequence` 后又是 `(2, 7, 512)`。**PAD 回来了。**

##### 6.2 两个关键 PAD 编号（**不能是 0**）

```python
PAD_TOKEN_EN = english_words_count    # 2803   ← 词表 0~2802 之后的第一个空位
PAD_TOKEN_FR = french_words_count     # 4345
```

| 编号 | 是什么 |
| --- | --- |
| 0 | **SOS** ← 已被占 |
| 1 | **EOS** ← 已被占 |
| 2 ~ 2802 | 真实词 |
| **2803** | **PAD** |

**为什么不能用 0**：`nn.Embedding(..., padding_idx=0)` 会把第 0 行**冻结成 0 向量且永远不给梯度**。SOS 也是 0 → SOS 被一起冻死，解码器第一步就喂进一个全 0 垃圾（实测梯度 `[0,0,0,0]`）。

**合法 PAD 编号的两个条件**：① 没人用过；② `< Embedding 行数`（否则 `IndexError`）。所以模型要 `+1` 行。

##### 6.3 ★ 函数速查（作用 + 参数 + 返回）

**`my_collate(batch)`** —— 把 N 条长度不一的样本拼成一个 batch（`DataLoader` 的 `collate_fn`）

| 参数 | 类型 / 形状 | 含义 |
| --- | --- | --- |
| `batch` | `list`，长度 = `batch_size` | 每一项是 `__getitem__` 的返回值，形如 `([2,3,4,1], [2,3,4,5,1])` |

**返回** `(x, y, x_len, y_len)`：

| 返回项 | 形状 | 含义 |
| --- | --- | --- |
| `x` | `(B, L_en)` long, **GPU** | 英文编号，短的末尾补 `PAD_TOKEN_EN` |
| `y` | `(B, L_fr)` long, **GPU** | 法文编号，同上 |
| `x_len` | `(B,)` long, **CPU** | 英文真实长度（★ 必须 CPU，`pack` 的硬要求） |
| `y_len` | `(B,)` long, **CPU** | 法文真实长度（当前用 `ignore_index` 代替，暂未使用） |

> `L_en` / `L_fr` 取**这个 batch 内的最长**，不是全局 `MAX_LENGTH`（省显存）。
> 传参时给**函数名不加括号**：`collate_fn=my_collate` ✓ / `my_collate()` ✗

**`My_Dataset.__getitem__(idx)`** —— 返回**两个 list**（不是张量），长度可以不同

**`my_dataloader(batch_size=1, shuffle=True)`** —— `batch_size` 一批几句、`shuffle` 是否打乱；返回 `DataLoader`（内部已挂 `collate_fn=my_collate`）

**`EncoderRNN(input_size, hidden_size, embd_dim=256, pad_idx=None)`**

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `input_size` | **2804** = `english_words_count + 1` | ★ `+1` 是给 PAD 多开一行 |
| `hidden_size` | 512 | GRU 隐藏维度 |
| `embd_dim` | 256 | 词向量维度 |
| `pad_idx` | **2803** | PAD 编号 → 那行不学梯度 |

**`EncoderRNN.forward(input, lengths)`**

| 参数 | 形状 | 含义 |
| --- | --- | --- |
| `input` | `(B, L)` long | 补齐的编号 |
| `lengths` | `(B,)` long, **CPU** | 真实长度 |

**返回** `(output, hidden)`：

| 返回项 | 形状 | 用途 |
| --- | --- | --- |
| `output` | `(B, L, 512)` | 解码器的 **K 和 V**（PAD 位置填 0） |
| `hidden` | `(1, B, 512)` | 解码器的**初始状态**（"中间语义"） |

> 内部流程：`embedding` → **`pack_padded_sequence`** → `gru` → **`pad_packed_sequence`**
> `enforce_sorted=False`：不用自己按长度降序排，PyTorch 排完会**自动还原顺序**
> `total_length=input.size(1)`：让输出长度恒定，不随 batch 内容飘

**`Decoder_Attention(hidden_size, output_size, embd_dim=256, pad_idx=None)`**

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `output_size` | **4346** = `french_words_count + 1` | ★ 同样 `+1` 给 PAD |
| `pad_idx` | **4345** | |

**`Decoder_Attention.forward(input, hidden, encoder_outputs, enc_mask=None)`**

| 参数 | 形状 | 含义 |
| --- | --- | --- |
| `input` | `(B, 1)` long | **当前这一步**的目标词（`SOS` 开头） |
| `hidden` | `(1, B, 512)` | 解码器状态，**充当 Q** |
| `encoder_outputs` | `(B, L, 512)` | **K 和 V** |
| `enc_mask` | `(B, L)` bool 或 `None` | `True`=真词，`False`=PAD |

**返回** `(output, hidden, attn_weights)`：

| 返回项 | 形状 | 含义 |
| --- | --- | --- |
| `output` | `(B, 4346)` | 每个样本的词表分布（★ 必须 `out(output.squeeze(1))`，不能 `output[0]`） |
| `hidden` | `(1, B, 512)` | 更新后的状态 |
| `attn_weights` | `(B, 1, L)` | 注意力权重，和 = 1（PAD 位精确 0） |

**`one_train(encoder_rnn, decoder_rnn, encoder_optimizer, decoder_optimizer, criterion, x, y, x_len)`** —— 一次前向 + 一次参数更新

| 参数 | 形状 | 含义 |
| --- | --- | --- |
| `x` | `(B, L_en)` | 英文编号 |
| `y` | `(B, L_fr)` | 法文编号 |
| `x_len` | `(B,)` | 英文真实长度 |
| `criterion` | — | 必须 `CrossEntropyLoss(ignore_index=PAD_TOKEN_FR)` |

**返回** `float`：平均 loss（`sum / y.shape[1]`）

> 内部步骤：`zero_grad` → 推 `enc_mask` → 编码 → **循环时间步** → `loss/L` → `backward` → `clip` → `step`

**`train(total_epochs=None, resume_from=0)`** —— 训练主循环

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `total_epochs` | `None` → 用全局 `epoch`（10） | 总共训到第几轮 |
| `resume_from` | `0` | `0`=从头训；`12`=加载 `model/*_12.pth` 接着训，从第 13 轮开始存 |

> 内部结构：两重循环（轮 / 步）+ 三个累加器（打印、记点、整轮）+ 每轮存档。
> **断点续训三行**：建模型 → `load_state_dict` 灌权重 → `for e in range(total_epochs - resume_from)`，
> 存档名用 `real_epoch = resume_from + e + 1`（★ 不然会覆盖旧文件）。
>
> ⚠️ **已知限制**：`loss_list` / `epoch_losses` 是**每次 `train()` 调用独立**的变量 →
> 续训时曲线只覆盖"这次续训的那一段"（日志里的轮次是真轮次，但图上从 0 开始）。
> 例：`train(total_epochs=50, resume_from=10)` 画出来的右图只有 **40 个点（第 11~50 轮）**。

```python
train()                                    # 从头训 10 轮
train(total_epochs=30)                     # 从头训 30 轮
train(total_epochs=50, resume_from=12)     # 加载第 12 轮，训到 50 轮
```

> 💡 **权重文件会一直堆**：每轮 27.5 MB，训 50 轮就是 **1.35 GB**。
> 只要最终模型的话，训完保留 `encoder_rnn_50.pth` / `decoder_rnn_50.pth`，其余删掉即可
> （`model/` 已在 `.gitignore` 里，删除不影响仓库）。
> **删之前先验证留下的那个能加载、能翻译** —— 别删完了才发现留下的是坏的。

**`evaluate(sentence, epoch=5)`** —— `sentence` 英文句子、`epoch` 用第几轮的权重；返回法文串（单句推理，无 PAD，`enc_mask` 传 `None`）

##### 6.4 ★ 变量名速查（这批名字最容易搞混）

| 名字 | 形状 | 到底是什么 |
| --- | --- | --- |
| `x` / `y` | `(B, L)` | 英文 / 法文**编号**（不是词向量） |
| `x_len` / `y_len` | `(B,)` | 真实长度（CPU） |
| `enc_mask` | `(B, L)` bool | 从 `x != PAD_TOKEN_EN` **现算**出来的 |
| `enc_out` / `encoder_outputs` | `(B, L, 512)` | 编码器**每个位置**的输出 = 注意力要的 K/V |
| `hidden` / `enc_hidden` | `(1, B, 512)` | 编码器**最后一个**状态 = 解码器初始状态 |
| `inp` / `input_y` | `(B, 1)` | 解码器**当前步**输入 |
| `out` / `decoder_output` | `(B, 4346)` | 当前步的词表分布 |
| `B` / `L` | 标量 | batch 大小 / 补齐后的长度 |

##### 6.5 实测数据

**正确性（改造没改变数学）** —— 同一份权重下，batch 里每条样本的 loss == 单独跑的 loss：

| 样本 | batch 里的 loss | 单独跑 | 一致? |
| --- | --- | --- | --- |
| 行0 | 8.420256 | 8.420251 | ✓ |
| 行1 | 8.350683 | 8.350674 | ✓ |
| 行2 | 8.358961 | 8.358958 | ✓ |

**pack 有效**：同一句"batch 内 hidden" vs "单独跑 hidden" → 差 **4.6e-05**；去掉 pack → 差 **4.0e-01**（差 1 万倍）

**mask 有效**：PAD 位置权重 **0.135 → 0.000**，真词权重和 **1.0000**

**速度**：`batch=32` 每句 0.64 ms vs `batch=1` 12.85 ms → **快 20 倍**；每轮 0.7 分钟（原 9.8 分钟）

**收敛**：前 300 步 loss **7.24 → 2.87**（降 60%）

##### 6.6 ★ 三个"不报错但悄悄变差"的坑

| 坑 | 症状 | 实测 |
| --- | --- | --- |
| **忘了 mask** | `hidden` 干净，但真词权重被稀释 | 5 词句补到 8 位，真词权重和只剩 **0.5568**；`context` 严格等于正确值 × 0.5568（**等比缩小**，不是被污染） |
| **`masked_fill` 忘了 `~`** | 权重和仍是 1，但全给了 PAD | `[1,2,3]` + mask `[T,T,F]` → `[-inf,-inf,3.0]`（把真词填死了） |
| **`unsqueeze(1)` 漏了** | **不报错**，但形状静默从 `(B,1,L)` 变成 `(B,B,L)` | 3 步之后在 `torch.cat` 才炸：`Expected size 1 but got size 3` |

##### 6.7 改造前后对照

| | 改造前 | 改造后 |
| --- | --- | --- |
| `__getitem__` 返回 | 张量 + `.to(device)` | **list**（长度可变） |
| 组 batch | 默认 `collate_fn`（`stack`）→ 长度不同就报错 | `my_collate` 补齐 + 给长度 |
| 编码器签名 | `forward(input, hidden)` | `forward(input, lengths)` |
| 解码器签名 | `forward(input, hidden, encoder_outputs)` | `+ enc_mask=None` |
| 取一步目标 | `y[0][i].view(-1)` → `(1,)` | `y[:, i]` → `(B,)` |
| 取预测 | `topi.item()` | `argmax(1)` + `finished.all()` |
| 输出层 | `self.out(output[0])` | `self.out(output.squeeze(1))` |
| loss | 累加后直接 backward | **先 `/L` 再 backward** |
| 损失函数 | `CrossEntropyLoss()` | `+ ignore_index=PAD_TOKEN_FR` |
| 模型尺寸 | `Embedding(2803, 256)` | `Embedding(2804, 256, padding_idx=2803)` |

> **⚠️ 旧 checkpoint 全部失效**（`Embedding` 从 2803 行变 2804 行），必须重训 —— 因为 `batch=32` 一轮只要 0.7 分钟，10 轮 7 分钟，代价可接受。

##### 6.8 ★ batch=32 的最终正确率（50 轮，已饱和）

同一 200 句测试集、同一"多参考匹配"口径：

| 轮次 | 10 | 11 | 12 | **15** | 20 | 25 | 30 | **35** | 40 | **45** | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 准确率 | 75.5% | 84.0% | 90.5% | **96.0%** | 96.5% | 98.0% | 97.0% | **98.5%** | 97.5% | **98.5%** | 97.5% |

**峰值在第 35 / 45 轮 = 98.5%，第 50 轮 97.5%。**

**★ 第 15 轮就到顶了**

从第 15 轮起一直在 **96.0% ~ 98.5%** 之间来回波动，**没有上升趋势**。
200 句样本在 97.5% 处的 95% 置信区间约 **±2.2%**（±4 句）→ **15 轮之后的变化全是采样噪声**。

> **训 15 轮（9 分钟）和训 50 轮（36 分钟）效果一样，后 35 轮基本是无用功。**

**★ 与 batch=1 的对比（这才是改造的价值所在）**

| 目标水平 | batch=1 | batch=32 | 加速 |
| --- | --- | --- | --- |
| ~92% | **49 分钟**（5 轮 92.5%） | **7 分钟**（12 轮 90.5%） | **7 倍** |
| ~96% | 未达到（5 轮停在 92.5%） | **9 分钟**（15 轮 96.0%） | — |
| 峰值 98.5% | 未测 | **25 分钟**（35 轮） | — |

> **改造的收益不是"同样轮次快 20 倍"，而是"同样时间能多训 7 倍轮次"。**
> 到饱和点之后，多训只是浪费时间 —— 所以"能训得快"的真正用处是**快速试错**（调 lr、比对方案），不是"堆轮次刷分"。

**★ 之前记录的两个失败案例都修好了**

| 案例 | 当时的表现 | 50 轮后 |
| --- | --- | --- |
| `i am from shizuoka .` | 前 4 轮全错（长尾词 `shizuoka` 退化成 `suis`），第 5 轮才修好 | ✅ `je suis de shizuoka .` |
| `you re out of luck .` | 第 10 轮错成 `tu chance est epuisee`（所有格错） | ✅ `votre chance est epuisee .` |

**剩余 2.5% 的错误**：① "一对多"语料（一句英文多个法语答案，模型只能给折中）；② 罕见词长尾；③ 超长句被 `max_len` 截断 —— 都是**数据固有难度**，不是模型 bug。

### Day07 英译法案例（下）+ Transformer 登场（P96–P109）

已全部看完 **P96–P109**。P96–P104 的内容在学这一段之前就已经自己实现过，所以是"回头对一遍"。

#### 1. P96–P104 与自己实现的对应关系

**这一段的代码在本阶段之前就已经自己写完了**，所以看课时相当于"回头对一遍"，标记为 `[x]`：

| 课程分集 | 内容 | 自己代码里的位置 |
| --- | --- | --- |
| P96 | 带注意力解码器测试代码 | `test_decoder_Attention`（打印 out / hidden / attn 三个形状，末位 PAD 权重为 0） |
| P97 | 模型搭建总结 | `EncoderRNN` + `Decoder_Attention` |
| P98 | **Teacher Forcing** | `teacher_forcing_ratio = 0.5`（按句子掷骰子） |
| P99 | 训练·单批次 | `one_train` |
| P101 | 训练·完整代码 | `train`（自己另加了 `resume_from` 断点续训） |
| P102 | 训练总结 | — |
| P103 | 预测·代码实现 | `evaluate`（free-running） |
| P104 | 预测·代码测试 | 200 句固定测试集，50 轮后峰值 **98.5%** |

**课程里没有、自己加的**：batch 化改造（pad + pack + mask），见上面 `#### 6.`。

#### 2. P105 张量图：为什么要画它

课程只是"了解"级别，但**这一步的价值在调试时最高**。本项目 `#### 6.4 变量名速查` 那张表本质上就是一张张量图：

```
x (B,L) → embedding (B,L,256) → pack (.data = Σlens, 256) → gru → (Σlens, 512)
        → unpack (B,L,512) → 注意力 score (B,1,L) → softmax → context (B,1,512)
        → cat (B,1,768) → attn_combine (B,1,512) → gru (B,1,512) → out (B,4346)
```

**收益**：这次改造中三个"不报错但静默出错"的问题（`masked_fill` 少取反、少 `unsqueeze(1)`、`output[0]` 只取第 0 个样本），**全都是靠对照形状链发现的** —— 形状图上"1 变 B""768 变 512"这种地方，一画就露馅。

#### 3. P106 Transformer 的由来：RNN 的三个硬伤

这里只是"了解"级别，但它是后面所有 Transformer 内容的动机，值得先记下来：

| RNN 的硬伤 | 具体表现 | 在本项目里的证据 |
| --- | --- | --- |
| **时序依赖 → 无法并行** | 第 t 步必须等第 t-1 步算完 | 解码器只能写 `for i in range(y.shape[1])` 逐步循环，batch=32 也只是"32 句一起串行"；**瓶颈在时间维，不在 batch** |
| **长距离依赖衰减** | 信息要一步步传递，早期内容被冲淡 | 无 Attention 的解码器只把编码器**最后一个** `hidden` 当中间语义（信息瓶颈）；加了注意力才解决 —— 这正好是 Day05/Day06 的对照实验 |
| **固定长度瓶颈** | 无论源句多长都压成一个定长向量 | 同上；注意力把"一个向量"换成"每个词一个向量 + 动态加权"，才是质变 |

> **一句话**：**RNN 是"串行 + 压成定长"，Transformer 是"并行 + 每步都能看到全部"** —— 自注意力就是把 Day05 学的注意力机制从"解码器→编码器"推广成"每个词都能和所有词算注意力"。

#### 4. P107–P109 Transformer 架构总览

三集是纯理论，但正好是后面 34 集的骨架（论文 Figure 1）：

```
源句 ──► 词嵌入 ──► + 位置编码 ──► 编码器 ×6 ─────────────────┐
                                                              │ memory
目标句 ─► 词嵌入 ──► + 位置编码 ──► 解码器 ×6 ──► Linear ──► 词表分数
                                    ▲                         │
                                    └─────────────────────────┘
                                      （每层用一次交叉注意力）
```

| 子层 | 编码器层 | 解码器层 | Q / K / V |
| --- | --- | --- | --- |
| 自注意力 | ✓ | ✓（**带因果掩码**） | 都是自己 |
| 交叉注意力 | ✗ | ✓ | Q = 解码器，K/V = 编码器输出 |
| 前馈网络 | ✓ | ✓ | — |

> **一句话**：**编码器负责"读懂原文"，解码器负责"边写边查原文"。**

---

### Day08 Transformer 逐层实现（上）：词嵌入 / 位置编码 / 注意力 / 前馈（P110–P125）

代码在 **`transformer架构/`**（曾误拼为 `transfomer架构`，已改名）：

| 文件 | 内容 | 对应分集 |
| --- | --- | --- |
| `dm01_input.py` | `Embedding`、`PositionalEncoding` | P111 / P114–P115 |
| `dm02_encoder_element.py` | `dm01_test_triu`、`attention`、`Multi_Head_Attention`、`PositionwiseFeedForward`、`LayerNorm` | P116–P125（+P126） |
| `trace_utils.py` | `trace`（树形打印层级 + 形状）、`param_summary`、`Wrap` | 自建工具 |

#### 1. 注意力四步与形状链

```
① 算分数   scores = Q·Kᵀ / √d_k
② 归一化   attn   = softmax(scores)
③ 加权求和 out    = attn · V
④ 融合     out    = W_o · out        （即 Multi_Head_Attention.linear_out）
```

**多头之后的完整形状链（B=2, L=10, d_model=512, h=8, d_k=64）：**

```
(B, L, 512) ─Linear→ ─view(B,-1,8,64)→ (B,L,8,64) ─transpose(1,2)→ (B,8,L,64)
Q·Kᵀ ─→ scores (B,8,L,L) ─softmax→ attn (B,8,L,L)
attn·V ─→ (B,8,L,64) ─transpose+contiguous+view→ (B,L,512) ─Linear→ (B,L,512)
```

| 细节 | 说明 |
| --- | --- |
| `transpose` 只改 stride，不改内存 | 后面要 `.view()` 必须先 `.contiguous()`（`.reshape()` 等价于两者组合） |
| `8 × 64 = 512` | 分头只是"切"，不是"压缩"，`reshape` 回 `d_model` 无额外计算 |

#### 2. 多头 vs 单头：参数量一样，差在"几个 softmax"

| | 单头 | 多头（h=8） |
| --- | --- | --- |
| 参数量 | 1,050,624 | 1,050,624（**完全相同**） |
| 计算量 | 相同 | 相同 |
| 注意力矩阵个数 | **1 个** `(L,L)` | **8 个** `(L,L)` |
| softmax 次数 | **1 次** | **8 次** |

**关键验证**：`Σ score_i = score_single × √8`（最大偏差 **1.43e-06**），但

```
softmax(Σ score_i) ≠ Σ softmax(score_i)      实测最大差 0.3299，平均差 0.0969
```

> **★ 不能"先把 8 个头的分数加起来再 softmax"** —— 那样就退化成单头了。
> **多头的价值不在参数量，而在"同时用 8 套不同的相似度度量看同一句话"。**

#### 3. 为什么必须有 `linear_out`（`W_o`）

| 理由 | 实测证据 |
| --- | --- |
| **8 个头量纲不同** | 各头输出 std 在 **0.094 ~ 3.157** 之间（差 **33 倍**）；残差 `x + MHA(x)` 要求同量纲 |
| **不加 `W_o` 时每个头只影响自己的 64 维** | 不加：只有 `64/512` 个维度变化；加了：`512/512` 全变 |

（`W_o` 若是块对角矩阵也会退化成"各管各的"，所以它必须是全连接混合。）

#### 4. 8 个头都看全部 512 维，但每个只输出 64 维

```python
self.linear_q = nn.Linear(512, 512)      # 一个矩阵
# 分头时 Wq[0:64, :] 就是第 0 个头的投影 (64, 512)
# 8 个 (64,512) 叠起来正好是 (512,512)
```

**实测：8 个独立的 `Linear(512→64)` 与 1 个 `Linear(512→512)` + 切分，输出与参数量完全一致。**

> **★ "多头"不是"把输入切成 8 份各看 64 维"，而是"8 套 `(64,512)` 投影，每套都看全部 512 维"。**

#### 5. 位置编码：为什么是 `x + pe` 而不是拼接

| 做法 | 维度 | 问题 |
| --- | --- | --- |
| 拼接 | `2×d_model` | 后面所有层维度都要改 |
| **相加** | `d_model` | ✓ 论文选这个：位置信息"叠加"在语义向量上，不改维度 |

**⚠️ 量级提醒**：加了 `×√d_model` 后，embedding 的 std ≈ **18.05**，位置编码 std ≈ **0.59** ——
**位置编码只占 3.3%**。它不"主导"语义，作用是给每个位置一个**可区分的微小偏移**。

---

### Day09 Transformer 逐层实现（下）：规范化 / 编码器 / 解码器 / 整体拼装（P126–P143）

| 文件 | 内容 | 对应分集 |
| --- | --- | --- |
| `dm02_encoder_element.py` | `LayerNorm`（自定义 `a_2`/`b_2`） | P126–P127 |
| `dm03_encoder_sublayer.py` | `SublayerConnection` | P129–P130 |
| `dm04_encoder_layer.py` | `EncoderLayer` | P131–P132 |
| `dm05_encoder.py` | `Encoder`（`clones` + 末尾 norm） | P133 |
| `dm06_decoder_layer.py` | `DecoderLayer`（3 个子层） | P134–P135 |
| `dm07_decoder.py` | `Decoder` | P136 |
| `dm08_output.py` | `Output`（Linear，**不接 softmax**） | P137–P138 |
| `dm09_transformer.py` | `Transformer` + `make_model` + `use_transformer` | P139–P141 / P143 |

#### 1. 三种注意力：一张表说清

| 注意力 | Q 来自 | K/V 来自 | 分数矩阵 | 掩码 |
| --- | --- | --- | --- | --- |
| **编码器自注意力** | `x` | `x`（同源） | **方阵** `(B,h,L_src,L_src)` | 只要 padding |
| **解码器自注意力** | `x` | `x`（同源） | **方阵** `(B,h,L_tgt,L_tgt)` | **因果 + padding** |
| **交叉注意力** | `x`（解码器） | `memory`（编码器） | **矩形** `(B,h,L_tgt,L_src)` | 只要 padding |

**交叉注意力为什么不要因果掩码**：K/V 是**已完整确定的源句**，"未来"这个概念只在"自己正在逐词生成"的那一侧才存在。

#### 2. 交叉注意力的维度要求

```
q (B,L_tgt,512) @ kᵀ (B,512,L_src) = scores (B,L_tgt,L_src)   ← 512 被吃掉
attn (B,L_tgt,L_src) @ v (B,L_src,512) = out (B,L_tgt,512)   ← L_src 被吃掉
```

| 维度 | 要求 |
| --- | --- |
| `B` | 三者必须相同 |
| `d_model` | Q/K/V 必须相同（投影层 `in_features` 都是它） |
| `L_tgt` / `L_src` | **可以不同** |
| `d_k` | Q 和 K 必须相同（点积要求） |

**实测**：`x (2,3,512)` + `memory (2,4,512)` → 输出 `(2,3,512)`，分数矩阵 `(2,8,3,4)`。

#### 3. ★ 掩码的三个坑

**坑 1：`tgt_mask` 用 `torch.ones` → 偷看未来**

把 `tgt` 最后一个词换掉，看位置 0 的输出变多少：

| 掩码 | 位置 0 差异 | 结论 |
| --- | --- | --- |
| `torch.ones(...)` | **0.111009** | ❌ 解码器看到了未来 |
| `tril(ones)` | **0.000000** | ✓ 因果正确 |

难发现的原因：不报错，训练 loss 还会异常地低（学会抄答案），推理时完全不会翻译。

**坑 2：`src_mask` 写成 `(B, L_src, L_src)` → `L_src ≠ L_tgt` 时崩**

同一个 `src_mask` 要喂**两个地方**：

| `src_mask` 形状 | 编码器自注意力 `(B,h,L_src,L_src)` | 交叉注意力 `(B,h,L_tgt,L_src)` | 通用？ |
| --- | --- | --- | --- |
| `(B, L_src, L_src)` | ✓ | ❌ 崩 | ✗ |
| `(B, L_tgt, L_src)` | ❌ 崩 | ✓ | ✗ |
| **`(B, 1, L_src)`** | ✓ | ✓ | **✓ 唯一通用** |

（demo 里 `(2,10,10)` 能跑，只因 `L_src == L_tgt == 10` 碰巧相等。）

**坑 3：掩码禁掉"整行" → NaN 扩散到全序列**

| 做法 | 结果 |
| --- | --- |
| 只禁 key 方向 `mask[:, :, pad] = 0` | ✓ 其他位置输出不变 |
| 连 query 行也禁 `mask[:, pad, :] = 0` | ❌ **全序列 NaN** |

**原因**：某行全 `-inf` → 该行 softmax = NaN → 下一层它作为 K/V → 所有位置都 NaN。
**PAD 位置自己那行不用禁**，它的输出是垃圾没关系，loss 里用 `ignore_index` 忽略即可。

#### 4. 完整模型与参数分布

```python
model = make_model(src_vocab=1000, tgt_vocab=1000, N=6, d_model=512, d_ff=2048, heads=8)
# 45,677,544 参数（174.25 MB float32）
```

| 部件 | 参数量 | 算式 |
| --- | --- | --- |
| 一个 `Multi_Head_Attention` | 1,050,624 | 4 × (512×512 + 512) |
| 一个 `PositionwiseFeedForward` | **2,099,712** | (512×2048+2048) + (2048×512+512) |
| 一个 `EncoderLayer` | 3,152,384 | 105 万 + 210 万 + 2×1,024 |
| 一个 `DecoderLayer` | 4,204,032 | 105 万×2 + 210 万 + 3×1,024 |
| 合计（N=6, vocab=1000） | **45,677,544** | — |

> **★ FFN 参数量是注意力的 2 倍** —— 很多人以为注意力是大头，其实 `512×2048×2` 更重。

#### 5. ★ `make_model` 最容易漏的一步：位置编码

```python
# ❌ 错：只有 embedding
src_embed = nn.Sequential(Embedding(src_vocab, d_model))

# ✅ 对：embedding + 位置编码打包
src_embed = nn.Sequential(Embedding(src_vocab, d_model),
                          PositionalEncoding(d_model, max_len=100, dropout=dropout))
```

**漏了会怎样**：Transformer 所有子层都**位置对称**（自注意力、FFN、LayerNorm、残差逐元素），
所以整个编码器满足 `enc(perm(x)) = perm(enc(x))`：

| `src_embed` 组成 | 换词序后输出最大差 | 能区分词序 |
| --- | --- | --- |
| 只有 `Embedding` | **0.000001** | ❌ 完全分不出 |
| `Sequential(Embedding, PositionalEncoding)` | **0.522509** | ✓ |

> **★ 这个 bug 不报错，只会让模型把"我爱你"和"你爱我"当成一回事。**

**自检代码（建议常备）：**

```python
ids = torch.tensor([[5, 7, 9, 11]]);  perm = torch.tensor([2, 0, 3, 1])
h1 = model.encoder(model.src_embed(ids))
h2 = model.encoder(model.src_embed(ids[:, perm]))
diff = (h2[0] - h1[0][perm]).abs().max()      # 应 > 1e-3
```

#### 6. 四项行为测试（能跑 ≠ 对）

| # | 测什么 | 方法 | 实测结果 |
| --- | --- | --- | --- |
| 1 | **因果性** | 改 `tgt` 末词，看位置 0 变不变 | 正确掩码 `0.000000` ✓ / 全 1 掩码 `0.111` ✗ |
| 2 | **位置编码** | 打乱 `src` 词序，看编码器输出变不变 | `0.2456` ✓ |
| 3 | **padding mask** | 改一个被 mask 的词，看其他位置变不变 | 只禁 key：不变 ✓；禁整行：NaN ✗ |
| 4 | **梯度** | `loss.backward()` 后算梯度范数 | `loss=3.4218`，范数 `2.4852` ✓ |

#### 7. 初始化：那三行 Xavier 要不要写

```python
for p in model.parameters():
    if p.dim() > 1:                    # 只挑 2 维以上（跳过 bias，bias 必须为 0）
        nn.init.xavier_uniform_(p)
```

**出处**：Annotated Transformer 原注释是 `# This was important from their code.` ——
**"their code" 指官方参考实现，不是论文附录**（论文正文只写了 Adam 的 β 和 warmup=4000）。

| 初始化 | Embedding std | `src_embed` 输出 std | **`encoder` 输出 std** | 150 步后 loss |
| --- | --- | --- | --- | --- |
| 不写 | 0.9694 | 8.24 | **0.9923** | 2.6289 |
| 写 | 0.1451 | 1.34 | **0.9923** | 2.6219 |

> **为什么差别小**：第一层 LayerNorm 会把整体尺度抹平，重要的只有词向量间的**相对结构**。
> **⚠️ 副作用**：`Embedding.weight` 也是 2 维，会被一起 Xavier 掉（vocab 大时 std 从 1.0 压到 0.0073，**小 137 倍**）。实践上仍能训（后面有 LayerNorm 兜着），学习阶段建议照抄。

#### 8. `SublayerConnection` 的对象树 ≠ 数据流

打印模块树会看到 `sublayers` 和 `self_attn` 是**兄弟**，看不出包裹关系 —— 因为它是"回调式"设计：

```python
x = self.sublayers[0](x, lambda t: self.self_attn(t, t, t, mask))
#                     ↑ 运行时才把 self_attn 传进去
```

```
【对象树】                          【数据流】
EncoderLayer                        x ──┬──────────────────────┐
├─ self_attn                            │ sublayers[0]         │ 残差
├─ feed_forward                         │   self_attn(...)     │
└─ sublayers (×2)                       │   + x → LayerNorm    │
   └─ norm, dropout                     └────────┬─────────────┘
                                        ┌────────┴─────────────┐
                                        │ sublayers[1]         │ 残差
                                        │   feed_forward(x)    │
                                        │   + x → LayerNorm    │
                                        └────────┬─────────────┘
```

#### 9. `trace_utils.py`：树形看结构

```python
from trace_utils import trace
trace(model, src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
```

同类型重复层自动折叠，37 行就能看完 6 层模型：

```
└─ model : Transformer    45,677,544
   ├─ encoder : Encoder  (2,10,512) → (2,10,512)    18,915,328
   │  ├─ layers : ModuleList    18,914,304
   │  │  └─ 0 : EncoderLayer  (2,10,512) → (2,10,512)    3,152,384   ×6 个
   │  │     ├─ self_attn : Multi_Head_Attention  ...    1,050,624
   │  │     │  └─ linear_q : Linear  (2,10,512) → (2,10,512)  ★ 262,656   ×4 个
   │  │     ├─ feed_forward : PositionwiseFeedForward  ...    2,099,712
   │  │     │  ├─ linear1 : Linear  (2,10,512) → (2,10,2048)  ★ 1,050,624
   │  │     │  └─ linear2 : Linear  (2,10,2048) → (2,10,512)  ★ 1,049,088
   │  │     └─ sublayers : ModuleList    2,048
   │  │        └─ 0 : SublayerConnection  ...    1,024   ×2 个
   │  └─ norm : LayerNorm  ★ 1,024
   └─ output_layer : Output  (2,8,512) → (2,8,1000)    513,000
```

**读法**：`├─`/`└─` 是层级；`(…) → (…)` 该模块的输入/输出形状；**★ = 参数就在这一层**。

**★ 还能直接看出"源句 10 词、目标句 8 词"** —— `encoder` 是 `(2,10,512)`，`decoder` 是 `(2,8,512)`，这正是交叉注意力要处理的长度差异。

#### 10. 疑问与澄清：学 Transformer 时的真实困惑

> 这一节记的是**我看课时实际问出来的问题**和当时的澄清，按出现顺序排列，每个都带验证过的数字。

| # | 我的疑问 | 澄清 | 关键证据 |
| --- | --- | --- | --- |
| 1 | 自注意力是"当前输入和之前输入的关系"，交叉注意力是"当前输入和编码器输出**那个包含所有信息的向量**的关系"？ | **前半对，后半要改**。编码器输出**不是"一个向量"**，而是 `L_src` 个向量（每个源词一个）。交叉注意力 = 解码器每个位置**逐个查阅** `L_src` 个源词 —— 这正是 Transformer 相对 RNN 没有信息瓶颈的地方 | `memory (2,4,8)` = 4 个源词各一个向量；分数矩阵 `(B,h,L_tgt,L_src)` 是**矩形** |
| 2 | 交叉注意力 K/V 是 `(B,L_src,d_model)`、Q 是 `(B,L_tgt,d_model)`，维度不一样不会报错吗？ | 只要 `B` 和**最后一维** `d_model` 相同就行。`q@kᵀ` 会把 `d_model` 吃掉，`attn@v` 会把 `L_src` 吃掉 | `(2,3,512)@(2,512,4)=(2,3,4)`；`(2,3,4)@(2,4,512)=(2,3,512)` |
| 3 | 交叉注意力还需要"不看后面"吗？两个词数不同的话，掩码不就变成 N×M 了吗？ | **完全想对了**。交叉注意力**不需要因果掩码**（源句完整，没有"未来"），掩码是 `(B,L_tgt,L_src)` 的矩形。**★ 但 `src_mask` 同时要喂编码器自注意力，所以唯一通用形状是 `(B,1,L_src)`** | `(B,L_src,L_src)` 只能喂编码器，`(B,L_tgt,L_src)` 只能喂交叉注意力；`L_src=8,L_tgt=5` 时前者直接崩 |
| 4 | 交叉注意力矩阵应该长什么样？ | **每一行都是满的**（每个目标位置都能看全部源词）。若出现"上三角全 0"，就是因果掩码被错用了 | 误用时位置 0 只看到源位置 0：`[1.000, 0, 0]`，丢掉 2/3 的信息 |
| 5 | `use_transformer` 跑通了是不是就没问题了？ | **能跑 ≠ 对**。forward 不报错只说明形状对，还得测四项：因果性 / 位置编码 / padding mask / 梯度 | 全 1 掩码下改末词，位置 0 输出变了 **0.111** → 偷看未来，**不报错** |
| 6 | Encoder 也要 mask？它又不"生成"，挡什么？ | Encoder 需要的是 **padding 掩码**，不是因果掩码。batch 里句长短不一要补齐，补出来的 PAD 是垃圾：不 mask 时 PAD 分走约 **40%** 注意力，且 PAD 位置会产出垃圾向量被下一层真实词看到 | `L=5` 含 2 个 PAD 时，真实词注意力占比只剩 **58.5%** |
| 7 | `forward` 已经走完全程了，为什么还要单独定义 `encoder`/`decode` 两个函数？ | 不重复，是分层封装。**推理时编码器只跑一次**（`memory` 缓存），再反复调解码器生成每个词；只有 `forward` 的话每步都要重跑编码器 | 生成 10 个词：每步重跑 **0.89 ms** vs 编码一次复用 **0.11 ms**（慢 **8.1 倍**） |
| 8 | 我都传了实体 `encoder` 了，直接 `self.encoder(...)` 不就好了？为什么还要 `def encoder`？ | **我的直觉是对的**。课程那层包装**撞名**了：`nn.Module` 把模块存进 `_modules`（不进 `__dict__`），同名方法胜出 → `self.encoder` 指向方法自己 → **无限递归** | `type(m.encoder) = method`；`RecursionError`。而 `decode` 没事（属性叫 `decoder`，多一个 `r`） |
| 9 | 论文是这样写的吗？但我这样写**没有位置编码**吧？ | 论文确实把位置编码加在 embedding 之后（实现里打包进 `src_embed`）。**而且我确实漏了** —— `PositionalEncoding` 定义了却从没被调用 | 只有 `Embedding`：换词序输出差 **0.000001**（数学上相等）；有位置编码：**0.522509** |
| 10 | `nn.init.xavier_uniform_` 那三行是干嘛的？论文附录 B 写的吗？ | 把 2 维以上的权重矩阵重新随机（`dim()>1` 是为了**跳过 bias**，bias 必须为 0）。**出处不是论文附录**，是官方参考实现（原注释 `# This was important from their code.`） | 副作用：`Embedding.weight` 也是 2 维会被一起 Xavier，std 从 1.0 压到 **0.0073**（小 137 倍） |
| 11 | 那**不写**那三行有问题吗？ | 小模型/学习阶段差别很小（150 步后 loss 2.6289 vs 2.6219）。原因：第一层 LayerNorm 会把整体尺度抹平 | `encoder` 输出 std 两边都是 **0.9923** |
| 12 | 打印出来的模块树看不懂，到底谁是谁的分支？ | `sublayers` 和 `self_attn` 在对象树上是**兄弟**，因为 `SublayerConnection` 是"回调式"设计（`forward(x, sublayer)` 接收函数），包裹关系只在调用里，不在对象树上 | `x = self.sublayers[0](x, lambda t: self.self_attn(t, t, t, mask))` |
| 13 | docstring 悬停显示的内容为什么比官方的少？ | 官方 `math.sqrt` 有**摘要行**（`Return the square root of x.`）。我的 docstring 一上来就是 `:param:`，没有摘要行 → 悬停只剩 `Parameters` 面板 | 规则：摘要行 → 空行 → 详细说明 → 空行 → `:param:` / `:return:` |

#### 11. ⇩ Transformer 还缺什么（课程 P143 之后不再讲，留待空闲补）

**课程讲完架构就跳到 Fasttext 了，全程没讲怎么训练这个 Transformer。** 目前 `dm09` 只能 forward，还不能翻译句子。

> **⚠️ 先明确一点：`dm09.forward` 本身是对的，不用改。** 下面的 1/3/4/5 是**新增文件**，
> 第 2 项是**可选重构**（实测不加也能写推理，见下面备注）。

| # | 缺什么 | 作用 | 备注 |
| --- | --- | --- | --- |
| 1 | `make_pad_mask` / `make_causal_mask` / `make_tgt_mask` | 三个掩码构造函数 | ★ `src_mask` 必须是 `(B,1,L_src)`；`tgt_mask` 用 `tril` |
| 2 | *(`encode` / `decode` 方法)* | **可选，不是必须**。只是让推理代码短一点 | 实测不加也能写 `greedy_decode`：直接 `model.encoder(model.src_embed(src), src_mask)` 即可。⚠️ 若加，名字别叫 `encoder`（见踩坑 15） |
| 3 | `greedy_decode` | 自回归生成：从 SOS 起，每步取 argmax，遇 EOS 停 | 推理入口；**依赖 `forward` 已有的部件，不需要改 `dm09`** |
| 4 | `SimpleLossCompute` + 训练循环 | `CrossEntropyLoss(ignore_index=PAD)` + Adam + warmup | 复用 `英译法/` 的 `my_collate` 与词表 |
| 5 | 数据管道 | `英译法/data/eng-fra-v2.txt`（63,594 句对）+ 词表 | EN **2803** / FR **4345** |

**`greedy_decode` 的最小写法（已实测跑通，不需要给 `Transformer` 加任何方法）：**

```python
@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len=50, sos=0, eos=1):
    B = src.size(0)
    memory = model.encoder(model.src_embed(src), src_mask)      # ★ 只算一次，缓存
    ys = torch.full((B, 1), sos, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool)
    for _ in range(max_len):
        L = ys.size(1)
        tgt_mask = torch.tril(torch.ones(L, L)).repeat(B, 1, 1)  # 因果
        h = model.decoder(model.tgt_embed(ys), memory, src_mask, tgt_mask)
        nxt = model.output_layer(h)[:, -1].argmax(-1)            # 取最后一步的 argmax
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        done |= nxt.eq(eos)
        if done.all():
            break
    return ys
```

**参考规模**：`d_model=512, N=6, heads=8, d_ff=2048` → 45.7 M 参数（vocab=1000）；换成实际词表约 **48 M**。
在 RTX 5060 Ti 上可与 `英译法/` 的 GRU 版做对照实验（GRU 版 batch=32 峰值 **98.5%**）。

## 环境

| 项 | 版本 / 说明 |
| --- | --- |
| 操作系统 | Windows |
| 项目环境 | **`nlpbase`**（conda）｜`D:\Anaconda3\envs\nlpbase\python.exe` |
| Python | 3.10.21（conda-forge） |
| TensorFlow | **2.19.0**（Day01 的 one-hot 用的是 `tensorflow.keras` 的 Tokenizer） |
| PyTorch | **2.14.0+cu130**（GPU 版，2026-10-10 装入；配套 `torchvision 0.29.0+cu130`） |
| TensorBoard | 2.19.0（随 TensorFlow 装，词向量可视化用） |
| fastText | **fasttext-wheel 0.9.2**（Windows 上免编译） |
| NumPy | 2.1.3 |
| jieba | 0.42.1 |
| joblib | 1.5.1（保存 / 加载 tokenizer） |
| setuptools | 81.0.0（⚠ 见踩坑记录 4：新版移除了 `pkg_resources`，会让 TensorBoard 起不来） |
| IDE | **VSCode**（主力；`.vscode/` 已 gitignore）、PyCharm 2026.1（`.idea/` 已 gitignore） |

> **为什么单独开一个环境**：上一个深度学习项目用的 `base`（`C:\ProgramData\anaconda3`，Python 3.14.6 + torch 2.14.0+cu130）
> Python 版本太新，而 NLP 这套要装 TensorFlow / fasttext / transformers，这些库在 **3.10** 上踩坑最少。
> `base` 那套没动，随时能切回去；`learn` 环境里只有 jieba，暂时用不上。
>
> **切换解释器**：
> - **VSCode**：`Ctrl+Shift+P` → `Python: Select Interpreter` → 选 `D:\Anaconda3\envs\nlpbase\python.exe`（选一次即写入 `.vscode/settings.json`）
> - **PyCharm**：Settings → Project: nlp → Python Interpreter → Add Interpreter → Conda Environment → **Use existing** → 选 `nlpbase`

### 依赖现状（2026 年排查，基于 `nlpbase`）

**已装**：`tensorflow 2.19.0`、`torch 2.14.0+cu130`、`torchvision 0.29.0+cu130`、`tensorboard 2.19.0`、`fasttext-wheel 0.9.2`、`numpy 2.1.3`、`jieba 0.42.1`、`joblib 1.5.1`

课程接下来会用到、目前**还没装**：

| 依赖 | 从哪里开始需要 | 备注 |
| --- | --- | --- |
| `pandas` | Day02 文本分析（P26 起：标签分布、句子长度） | 还没装 |
| `matplotlib` | Day02 句子长度分布、散点图、词云（P28–P32） | 记得配 `SimHei` 中文字体 |
| `wordcloud` | Day02 高频形容词词云（P32） | 还没装 |
| `scikit-learn` | 后面的评估、切分数据 | 还没装 |
| `transformers` `datasets` | Day10–Day12 预训练模型与迁移学习 | 体积大，用到再装 |
| `opencc` | 繁转简（Day01 提过） | 可选 |

按课程进度**用到再装**，不要一上来全装 —— 环境干净，出问题好定位。

#### GPU 版 PyTorch（✅ 2026-10-10 已装好，命令留档备查）

国内直连 `download.pytorch.org` 只有几 kB/s，走阿里云的 wheel 镜像（**扁平目录，要用 `-f` 不是 `-i`**）：

```powershell
$py = 'D:\Anaconda3\envs\nlpbase\python.exe'
& $py -m pip install "torch==2.14.0+cu130" "torchvision==0.29.0+cu130" -f https://mirrors.aliyun.com/pytorch-wheels/cu130/ --timeout 120 --retries 10
```

如果提示找不到 cp310 的 wheel，把版本号去掉，让 pip 自己挑一个 cu13x 的版本。

装完做**三步验证**，第三步才是关键：

```python
import torch
print(torch.__version__, torch.version.cuda)   # 版本
print(torch.cuda.get_device_name(0))           # 显卡名
a = torch.randn(1000, 1000, device='cuda')     # ← 真的算一次
print((a @ a).sum().item())
```

前两步只说明「识别到了显卡」，**只有第三步跑通才证明 wheel 里编了 sm_120 的 kernel**，否则会报 `no kernel image is available`。

## 目录约定

代码按模块分目录，和课程的小节对应：

```
nlp/
├── jieba/                     Day01 P6–P12  分词、自定义词典、词性标注
│   ├── 01_jieba使用.py                        精确 / 全 / 搜索引擎三种模式 + load_userdict
│   ├── 02_POS（词性标注）.py                  pseg.lcut，加词典前后对比
│   └── data/use.txt                           自定义词典：电子科技大学 5 n / 软件工程 2 nz
├── 文本张量表示/               Day01 P12–P16 one-hot ｜ Day02 P19–P24 词向量
│   ├── 01_one-hot.py                          用 tensorflow.keras 的 Tokenizer 取 one-hot，joblib 存读
│   ├── 02_CBOW和skiggram.py                   Day02 P19–P22 fastText 训练 / 保存 / 加载 CBOW 与 SkipGram
│   ├── 03_word_embedding.py                   Day02 P24 nn.Embedding + TensorBoard add_embedding
│   ├── data/ai20ag、data/fil9                 fastText 训练语料（大文件，已 gitignore）
│   └── model/                                 tokenizer.pkl（小，提交）｜ CBOW/SkipGram_model.bin（972 MB，已 gitignore）
├── 文本分析/                   Day02 P26–P34 文本分析 ｜ Day03 P36 文本长度规范
│   ├── 01_标签数据分布.py                      P26 标签分布 countplot、P28 句长分布、P29 正负样本散点
│   ├── 02_n_gram实现.py                       P34 n-gram 文本特征
│   ├── 03_文本长度规范.py                      P36 按分位数定截断长度
│   └── data/train.tsv、dev.tsv、SimHei.ttf    影评数据 3000 + 1000 条（已提交）｜ SimHei 中文字体（已 gitignore）
├── RNN模型/                    Day03 P41 RNN ｜ P44–P49 LSTM / GRU ｜ Day03–04 P50–P65 人名分类案例
│   ├── 01_RNN.py                              P41 nn.RNN 的形状演示（input/output/hn）
│   ├── 02_LSTM.py                             P47 nn.LSTM + h0/c0，output/h1/c1 形状
│   ├── 03_GRU.py                              P49 nn.GRU（注意：只有 h，没有 c）
│   ├── 04_人名国家识别.py                       P50–P65 完整案例：Dataset/DataLoader + 三模型 + 训练 + 日志 + 画图 + 预测
│   ├── LSTM反向传播.md                         Day03 LSTM 反向传播笔记（课程没展开的部分，文末附复现代码）
│   ├── GRU门控与反向传播.md                     Day03 GRU 两个门的细节 + 反向传播（含四方衰减实测）
│   ├── data/name_classfication.txt            人名数据 20,074 条 / 18 个国家（312 KB，已提交）
│   ├── model/                                 my_rnn / my_lstm / my_gru.pth（共 782 KB，已提交）
│   └── log/                                   rnn/lstm/gru.json 训练日志 + 4 张对比图
├── 注意力机制/                  Day05 P66–P79 注意力机制（QKV / 权重 / 代码实现）
│   ├── 01_代码实现.py                          P77 注意力代码实现 —— 两个版本对照：
│   │                                           · My_Attention（位置打分版，课程/tutorial 写法）
│   │                                           · My_Attention_Content（内容比对版，K 是每个词一个）
│   └── 注意力步骤动画.html                     13 步交互动画：Seq2Seq+Attention 从编码到生成
│                                               （单文件、零依赖，双击浏览器打开；← → 翻步）
├── 英译法/                      Day06 Seq2Seq 英译法案例（batch=32 后峰值 98.5%，15 轮即饱和）
│   ├── translate.py                            数据清洗/预处理 → Dataset + **my_collate（补齐）**
│   │                                           → EncoderRNN（pack/unpack 跳过 PAD）
│   │                                           → DecoderRNN_No_Attention（无注意力，课程 P91）
│   │                                           → Decoder_Attention（有注意力 + mask，自己实现）
│   │                                           → one_train（按时间步循环）/ train / evaluate
│   ├── data/eng-fra-v2.txt                     英法句对 63,594 行（3.2 MB，已提交）
│   └── model/                                  encoder/decoder_rnn_**50**.pth（只留最终轮 27.5 MB，已 gitignore；
│                                               1~49 轮共 1.35 GB 已清理，省 1.3 GB）
├── transformer架构/              Day08–Day09 P110–P143 Transformer 逐层实现（完整模型 45,677,544 参数）
│   ├── dm01_input.py                           词嵌入层 Embedding（×√d_model）+ 位置编码 PositionalEncoding（register_buffer）
│   ├── dm02_encoder_element.py                 dm01_test_triu / attention（bmm+scale+softmax+dropout）
│   │                                           / Multi_Head_Attention（4×Linear, 1,050,624 参数）
│   │                                           / PositionwiseFeedForward（512→2048→512, 2,099,712）
│   │                                           / LayerNorm（自定义 a_2 / b_2）
│   ├── dm03_encoder_sublayer.py                 SublayerConnection：norm(x + dropout(sublayer(x)))（Post-LN）
│   ├── dm04_encoder_layer.py                    EncoderLayer（自注意力 + FFN，单层 3,152,384 参数）
│   ├── dm05_encoder.py                          Encoder（clones(layer, N) + 末尾 LayerNorm）
│   ├── dm06_decoder_layer.py                    DecoderLayer（因果自注意力 + 交叉注意力 + FFN，单层 4,204,032）
│   ├── dm07_decoder.py                          Decoder
│   ├── dm08_output.py                           Output（Linear(d_model, vocab)，直接返回 logits，不接 softmax）
│   ├── dm09_transformer.py                      Transformer 类 + make_model()（位置编码打包进 src_embed/tgt_embed）
│   │                                            + use_transformer() 端到端自测
│   └── trace_utils.py                           trace（树形打印模块层级 + 输入/输出形状 + 参数，自动折叠 ×N）
│                                                / param_summary / Wrap
├── runs/                      TensorBoard 事件文件（生成物，已 gitignore）
├── __pycache__/               Python 字节码缓存（生成物，已 gitignore）
├── .gitignore                 挡大文件 + runs/ + __pycache__
├── README.md
└── 课程进度.md                 189 P 完整课程地图与进度清单
```

后续目录（学到再建）：`Transformer/` Day07–09 · `Fasttext/` Day10 · `预训练模型/` Day10–12

## 分集 → 练习文件

| 分集 | 内容 | 练习文件 |
| --- | --- | --- |
| P2–P5 | NLP 概念 / 发展史 / 应用场景 / 文本预处理模块 | 纯理论，无代码 |
| P6 | 创建沙箱并关联 PyCharm | 环境配置（实际用的是 conda 环境 `nlpbase`），无代码 |
| P7 | jieba 精确模式 | `jieba/01_jieba使用.py` |
| P8 | jieba 全模式 | `jieba/01_jieba使用.py` |
| P9 | jieba 搜索引擎模式 | `jieba/01_jieba使用.py` |
| P10 | jieba 繁体字 | 未写（jieba 词典是简体，繁体要先 opencc 转简体再切） |
| P11 | 用户自定义词典 | `jieba/01_jieba使用.py`、`jieba/data/use.txt` |
| P12 | 文本预处理 NER 与 POS | `jieba/02_POS（词性标注）.py` |
| P13–P16 | one-hot（介绍 / 获取 / 使用 / 简单实现） | `文本张量表示/01_one-hot.py` |
| P17 | word2vec CBOW 原理 | 纯理论，无代码（要点见上面 Day01 第 4 节） |
| P18 | Word2Vec 之 SkipGram 词向量解释 | 纯理论（要点见 Day01 第 4 节；Skip-gram 更新次数是 CBOW 的 `2×window` 倍） |
| P19 | FastText 环境搭建 | `文本张量表示/02_CBOW和skiggram.py`（用 fasttext-wheel） |
| P20 | FastText 词向量模型的训练和保存 | `文本张量表示/02_CBOW和skiggram.py` |
| P21 | FastText 加载 / 查看 / 模型效果检验 | `文本张量表示/02_CBOW和skiggram.py` |
| P22 | FastText 模型参数设定 | `文本张量表示/02_CBOW和skiggram.py`（epoch / lr / dim / ws / minCount） |
| P23 | WordEmbedding 和 Word2Vec 的区别 | 纯理论 |
| P24 | WordEmbedding 获取词向量（+ TensorBoard 可视化） | `文本张量表示/03_word_embedding.py` |
| P25 | WordEmbedding 查看词向量的相似性 | `文本张量表示/02_CBOW和skiggram.py`（dm03：`get_nearest_neighbors`） |
| P26 | 文本分析_标签数据分布查看 | `文本分析/01_标签数据分布.py`（dm01） |
| P27 | 扩展_map() 函数功能介绍 | 纯理论（`map(f, it)` 逐个加工，返回**迭代器**，要 `list()` 才看得到） |
| P28 | 文本分析_句子长度分布分析 | `文本分析/01_标签数据分布.py`（dm02） |
| P29 | 文本分析_正负样本长度散点分布 | `文本分析/01_标签数据分布.py`（dm03 stripplot） |
| P30 | 扩展_chain() 函数介绍 | 纯理论（`chain(*nested)` 拉平一层嵌套，配 `Counter` 统计词频） |
| P31 | 文本分析_词汇统计 | 未落文件（已实跑：train 集 156,963 词 / 12,160 个不同词） |
| P32 | 文本分析_高频形容词词云 | 未写（`wordcloud` 已装，`01_标签数据分布.py` 里 import 了但没实现） |
| P33 | 扩展_zip() 函数用法解释 | 纯理论（并行取，和 map/chain 同族但管的是"配对"） |
| P34 | 文本特征处理_n-gram 代码演示 | `文本分析/02_n_gram实现.py` |
| P35 | 扩展_如何计算两个文本的相似度 | 未写 |
| P36 | 文本长度规范_代码实现 | `文本分析/03_文本长度规范.py` |
| P37 | 回译数据增强法介绍 | 纯理论 |
| P38–P40 | RNN 模型简介 / 分类介绍 / 模型结构 | 纯理论（要点见上面 Day03 笔记第 1 节） |
| P41 | RNN 代码_基础版 | `RNN模型/01_RNN.py` |
| P42–P43 | RNN 代码_修改句子长度 / 修改隐藏层及总结 | 未写 |
| P44–P45 | LSTM 模型_原理图解（上 / 下） | 理论 + 笔记 → `RNN模型/LSTM反向传播.md` |
| P46 | Bi-LSTM 模型_介绍 | 纯理论 |
| P47 | LSTM 模型_代码实现 | `RNN模型/02_LSTM.py` |
| P48 | GRU 模型_原理图解 | 纯理论（要点见上面 Day03 第 5 节） |
| P49 | GRU 模型_代码演示 | `RNN模型/03_GRU.py` |
| P50 | RNN 人名分类案例_需求介绍 | 纯理论（18 个国家、变长人名 → 分类任务） |
| P51 | 人名分类案例_获取全局字母表和国家名 | `RNN模型/04_人名国家识别.py`（`all_letters` / `categorys`） |
| P52 | 人名分类案例_读取数据到内存 | `RNN模型/04_人名国家识别.py`（`read_data`） |
| P53 | 人名分类案例_构建数据集对象 | `RNN模型/04_人名国家识别.py`（`My_ClassDataset`，`__getitem__` 里做 one-hot） |
| P54 | 人名分类案例_构建数据加载器对象 | `RNN模型/04_人名国家识别.py`（`my_Dataloader`） |
| P55 | 扩展_LogSoftmax() 函数讲解 | 纯理论（等价于 CrossEntropyLoss，见上面 Day04 第 5 节） |
| P56 | 搭建 RNN 模型 | `RNN模型/04_人名国家识别.py`（`My_RNN`） |
| P57 | 测试 RNN 模型 | `04_人名国家识别.py` 的 `dm02` 思路：一次性送 vs 逐字符送，结果相同 |
| P58 | 搭建 LSTM 和 GRU 模型 | `RNN模型/04_人名国家识别.py`（`My_LSTM` / `My_GRU`） |
| P59 | 测试三种模型 | 已实测：三个模型输出都是 `(1, 18)`，都能算 loss |
| P60 | RNN 模型训练 | `04_人名国家识别.py`（`train_rnn`）→ `log/rnn.json` |
| P61 | LSTM 模型训练 | `04_人名国家识别.py`（`train_lstm`）→ `log/lstm.json` |
| P62 | GRU 模型训练 | `04_人名国家识别.py`（`train_gru`）→ `log/gru.json` |
| P63 | 三种模型可视化 | `04_人名国家识别.py`（`save_train_log` / `plot_train_logs`）→ `log/*.png` |
| P64 | RNN 模型预测 | `04_人名国家识别.py`（`name2tensor` / `predict`，三模型 Top3 对比） |
| P65 | 思路总结 | 见上面 Day04 全节 |
| P66 | 注意力机制介绍 | 纯理论（要点见上面 Day05 第 1 节） |
| P67 | 注意力机制_QKV 简介 | 纯理论（Q/K/V 是**同一输入 × 三个不同矩阵**；见 Day05 第 1 节） |
| P68 | 注意力机制_实现步骤介绍 | 见上面 Day05 第 2 节（四步：算分数 → softmax → 加权求和 → 融合） |
| P69 | Seq2Seq 架构_任务介绍 | 纯理论（编码器读源句 → 中间语义 → 解码器逐词生成） |
| P70 | Seq2Seq 架构_加入注意力机制 | `注意力机制/注意力步骤动画.html`（13 步交互演示） |
| P71 | 扩展1_普通的 Encoder_Decoder 框架 | 见上面 Day05 第 3 节（Q = 解码器上一步状态，K/V = 编码器输出） |
| P72 | 扩展2_加入注意力机制的 Encoder_Decoder 框架 | 见上面 Day05 第 3 节 |
| P73 | 扩展3_注意力概率分布的计算方式 | 见上面 Day05 第 2 节（加性 vs 点积两种算法） |
| P74 | 扩展4_软注意力 / 硬注意力 / 自注意力简介 | 见上面 Day05 第 4 节（自注意力 = Q/K/V 同源，分数是方阵） |
| P75 | 常见的注意力计算规则介绍 | 见上面 Day05 第 2 节 |
| P76 | 扩展_bmm() 函数简介 | 见上面 Day05 第 1 节（bmm / matmul / einsum 三者对比与坑） |
| P77 | 注意力机制_代码实现 | `注意力机制/01_代码实现.py`（`My_Attention`，课程"位置打分"版） |
| P78 | 注意力机制_测试代码 | `注意力机制/01_代码实现.py` 的 `__main__`（两个版本对照） |
| P79 | 注意力机制_参数解释 | 见上面 Day05 第 5 节（`word_count` 硬编码带来的长度约束） |
| P80 | 英译法案例_需求介绍 | 纯理论（英→法翻译，63,594 句对） |
| P81 | 如何配置 CUDA 环境及测试（上） | 纯理论（本机 RTX 5060 Ti 15.9 GB / sm_120，torch 2.14.0+cu130） |
| P82 | 如何配置 CUDA 环境_实操 | 纯理论，无代码 |
| P83 | 扩展_配置 CUDA 环境总结 | 纯理论，无代码 |
| P84 | 数据清洗函数 | `英译法/translate.py`（`normalize_string`：小写 + 标点前加空格 + 去非字母） |
| P85 | 数据预处理函数 | `英译法/translate.py`（`read_data`：英文词表 **2803** / 法文 **4345**，SOS=0 EOS=1） |
| P86 | 构建数据集对象 | `英译法/translate.py`（`My_Dataset`，`__getitem__` 把句子转成编号 + EOS） |
| P87 | 获取数据加载器对象 | `英译法/translate.py`（`my_dataloader`；⚠️ 变长句子 batch_size 只能为 1） |
| P88 | 构建基于 GRU 的编码器 | `英译法/translate.py`（`EncoderRNN`：Embedding(2803,256) + GRU(256,512)） |
| P89 | 测试基于 GRU 的编码器 | `英译法/translate.py`（`test_encoder`；输出 `(1,5,512)` / `hidden (1,1,512)`） |
| P90 | 无 Attention 解码器_思路分析 | 见上面 Day05 第 6 节（中间语义只在第 0 步注入一次 = 信息瓶颈） |
| P91 | 构建基于 GRU 解码器（无 Attention） | `英译法/translate.py`（`DecoderRNN_No_Attention`：Embedding(4345,256) + Dropout + GRU + Linear(512,4345)） |
| P92 | 测试基于 GRU 的无 Attention 解码器 | `英译法/translate.py`（`test_decoder_No_Attention`） |
| P93 | 带 Attention 的解码器·思路分析 | 视频未看，自己实现：Q = 解码器状态，K/V = 编码器输出（见 Day05 第 3 节） |
| P94 | 带 Attention 的解码器·代码实现（上） | `英译法/translate.py`（`Decoder_Attention`；`attn_combine(768→512)` 让 GRU 不变宽） |
| P95 | 带 Attention 的解码器·代码实现（下） | `英译法/translate.py`（batch=1 五轮 33.0% → 59.5% → 72.5% → 82.5% → **92.5%**） |
| P96 | 带注意力解码器·测试代码 | `英译法/translate.py`（`test_decoder_Attention`） |
| P97 | 模型搭建总结 | 见 Day06（`EncoderRNN` + `Decoder_Attention`） |
| P98 | Teacher Forcing | 见 Day05 第 4 节（`teacher_forcing_ratio = 0.5`） |
| P99 | 训练·单批次 | `英译法/translate.py`（`one_train`） |
| P100 | 扩展 `view()` 演示 | 纯理论 |
| P101 | 训练·完整代码 | `英译法/translate.py`（`train`，自己另加了 `resume_from` 断点续训） |
| P102 | 训练总结 | 见 Day06（loss 参考值 `ln(词表) ≈ 8.38`） |
| P103 | 预测·代码实现 | `英译法/translate.py`（`evaluate`，free-running） |
| P104 | 预测·代码测试 | 200 句固定测试集，50 轮后句子级正确率峰值 **98.5%** |
| P105 | 绘制张量图 | 见 Day07 第 2 节 |
| P106 | Transformer 由来 | 见 Day07 第 3 节（RNN 三个硬伤） |
| P107 | Transformer 架构介绍（文字版） | 见 Day07 第 4 节 |
| P108 | 架构图·上 | 见 Day07 第 4 节（编码器侧） |
| P109 | 架构图·下 | 见 Day07 第 4 节（解码器侧） |
| P110 | 学习 Transformer 的路线 | 纯理论（无代码） |
| P111 | 词嵌入层代码 | `transformer架构/dm01_input.py`（`Embedding`：`nn.Embedding` + 乘 `√d_model`） |
| P112 | 位置编码原理 | 见 Day08 第 5 节 |
| P113 | 位置编码总结 | 见 Day08 第 5 节 |
| P114 | 位置编码代码 | `transformer架构/dm01_input.py`（`PositionalEncoding`，`register_buffer('pe')`） |
| P115 | 位置编码测试 | `dm01_input.py`（修掉 `self.pe[:x.size(1)]` 切片错误，应切第 1 维） |
| P116 | 扩展·上三角矩阵 | `dm02_encoder_element.py`（`dm01_test_triu`；`triu(ones, diagonal=1)` = 被挡住的位置） |
| P117 | 扩展·下三角矩阵 | `dm02`（`tril(ones)` = 允许的位置；★ `diagonal=0` 会把对角线也挡掉 → NaN） |
| P118 | 扩展·掩码可视化 | 纯演示 |
| P119 | 扩展·`masked_fill()` | `dm02`（`attention` 里 `masked_fill(mask == 0, -inf)`；`mask==0` 等价 `~mask`） |
| P120 | 注意力机制代码 | `dm02`（`attention()`：`bmm` + scale + softmax + dropout） |
| P121 | 多头注意力原理图·上 | 见 Day08 第 2 节 |
| P122 | 多头注意力原理图·下 | 见 Day08 第 2 节 |
| P123 | 多头注意力代码 | `dm02`（`Multi_Head_Attention`，4 个 `Linear(512,512)` = **1,050,624** 参数） |
| P124 | 多头注意力测试 | `dm02`（形状链见 Day08 第 1 节） |
| P125 | 前馈全连接层 | `dm02`（`PositionwiseFeedForward`，512→2048→512 = **2,099,712** 参数） |
| P126 | 规范化层代码 | `dm02`（`LayerNorm`，自定义 `a_2`/`b_2`） |
| P127 | 规范化层测试 | `dm02`（`use_LayerNorm`，输出均值≈0 / std≈1） |
| P128 | 扩展·BN 和 LN 的区别 | 纯理论（BN 按特征维、LN 按样本维） |
| P129 | 子层结构代码 | `dm03_encoder_sublayer.py`（`SublayerConnection`，**Post-LN**，符合论文） |
| P130 | 子层结构测试 | `dm03`（★ 若 norm 只加在分支上，16 层后 std 涨到 4.34；正确写法稳定 1.00） |
| P131 | 编码器层代码 | `dm04_encoder_layer.py`（`EncoderLayer`：自注意力 + FFN，各套一层子层连接） |
| P132 | 编码器层测试 | `dm04`（`use_encoder_layer`，单层 **3,152,384** 参数） |
| P133 | 编码器代码及测试 | `dm05_encoder.py`（`Encoder`：`clones(layer, N)` + 末尾 `LayerNorm`） |
| P134 | 解码器层代码 | `dm06_decoder_layer.py`（`DecoderLayer`：`self_attn(x,x,x,tgt_mask)` + `src_attn(x,memory,memory,src_mask)`） |
| P135 | 解码器层测试 | `dm06`（`L_tgt=3, L_src=4` 跑通；交叉注意力是**矩形**分数矩阵） |
| P136 | 解码器代码及测试 | `dm07_decoder.py`（`Decoder`，每层多一个交叉注意力子层） |
| P137 | 输出部分代码 | `dm08_output.py`（`Output`：`Linear(d_model, vocab)`，★ **直接返回 logits，不接 softmax**） |
| P138 | 输出部分测试 | `dm08`（`use_output`；形状 `(B,L_tgt,512) → (B,L_tgt,vocab)`） |
| P139 | 完整架构搭建·上 | `dm09_transformer.py`（`Transformer` 类：`encoder`/`decoder`/`src_embed`/`tgt_embed`/`output_layer`） |
| P140 | 完整架构搭建·下 | `dm09`（`make_model`；★ 位置编码必须打包进 `src_embed`/`tgt_embed`，漏了模型分不出词序） |
| P141 | 架构测试·上 | `dm09`（`use_transformer`；端到端 `(2,10)` + `(2,8)` → `(2,8,1000)`，无 NaN） |
| P142 | 架构组件介绍 | 见 Day09 第 4 节（参数分布表） |
| P143 | 架构测试·下 | 见 Day09 第 6 节（四项行为测试：因果性 / 位置编码 / padding / 梯度） |

## 运行方式

```powershell
# 项目环境是 nlpbase，直接用它的解释器
D:\Anaconda3\envs\nlpbase\python.exe jieba\01_jieba使用.py

# 词向量可视化：先跑脚本写事件文件，再起 TensorBoard（--logdir 必须和脚本里的 log_dir 一致）
D:\Anaconda3\envs\nlpbase\python.exe 文本张量表示\03_word_embedding.py
D:\Anaconda3\envs\nlpbase\Scripts\tensorboard.exe --logdir=D:\ML_Project\nlp\runs

# 想让局域网 / 手机也能看（要过防火墙，且它没有登录认证）：
D:\Anaconda3\envs\nlpbase\Scripts\tensorboard.exe --logdir=D:\ML_Project\nlp\runs --host=0.0.0.0 --port=6006
```

TensorBoard 起好后浏览器开 <http://localhost:6006>，顶栏切到 **PROJECTOR** 看词向量投影图。

或在 VSCode 里打开对应文件按 `F5`（PyCharm 是 `Shift+F10`）。

> ⚠ **工作目录很重要**：现有脚本里用的是相对路径（`./data/use.txt`、`./data/eng-fra-v2.txt`、`./model/tokenizer.pkl`），
> 它们相对于**运行时的工作目录**，不是相对于脚本文件本身。
> VSCode / PyCharm 默认把 Working directory 设成**项目根目录**（`D:\ML_Project\nlp`），
> 那样 `./data/use.txt` 会被解释成 `D:\ML_Project\nlp\data\use.txt` → `FileNotFoundError`
> （实测：从项目根目录跑 `英译法/translate.py` 会报 `FileNotFoundError: './data/eng-fra-v2.txt'`）。
> 两个办法，任选一个：
> 1. 改运行配置：VSCode 在 `launch.json` 里设 `"cwd": "${fileDirname}"`；PyCharm 是 Run → Edit Configurations → **Working directory** 填 `$FileDir$`，一劳永逸；
> 2. 把脚本里的路径改成 `Path(__file__).parent / 'data' / 'use.txt'`，从此不挑工作目录。
>
> Windows 上 `python` 命令被 Microsoft Store 的应用别名占用时，用上面的完整路径。

## 踩坑记录

> 跟着课程实际撞到的问题按顺序记在这里（自己补充）。

### 1. PyCharm 的「还原提交」(Revert) 不是"撤销提交"，它会把文件**从磁盘删掉**

- 场景：提交里混进了 2 个 972 MB 的 `.bin` + 1 个 680 MB 语料，想撤回，于是在 Git 日志里点了「还原提交」
- 后果：Revert 生成的是**反向提交**，对"新增文件"的提交来说就是**删除文件** —— 训练好的模型和刚写的 `.py` 一起没了
- 救回来：内容还在 `.git` 里，`git reset --hard <提交>` 或 `git checkout <提交> -- <路径>` 即可还原（当时逐字节核对过 SHA-1）
- **正确做法**：

  | 情况 | 用什么 |
  | --- | --- |
  | 提交**没 push**，想反悔 | Git 窗口 → Log → 右键最新提交 →「**撤消提交**」＝ `git reset --soft HEAD~1` |
  | 想退回到更早的提交 | 右键 →「将当前分支重置到此处…」→ 模式选「**软**」（硬重置会丢本地改动） |
  | 提交**已经 push** 了 | 才用「还原提交」(Revert) |

### 2. GitHub 单文件 100 MB 硬限制

- 两个 `.bin` 各 972 MB、`fil9` 680 MB → **push 一定会被拒**（超 100 MB 直接拒绝，超 50 MB 会警告）
- 处理：`git rm --cached` 只从版本库移除（磁盘文件保留）+ 写 `.gitignore`
- 教训：**训练产物和语料不要进 git**；真要分享用 Git LFS / GitHub Release / 网盘

### 3. TensorFlow 在 Windows 中文路径下会崩（本项目最大的坑）

- 报错：`tensorflow.python.framework.errors_impl.FailedPreconditionError: . is not a directory`
  （来自 `SummaryWriter` → `EventFileWriter` → `tf.io.gfile.makedirs`，不是 PyTorch 的问题）
- 根因：**目录名是中文**（`文本张量表示`）。TensorFlow 的 Windows 文件层列举目录时返回的是 **ANSI(GBK) 字节**，却按 UTF-8 解码：

  ```
  tf.io.gfile.listdir(中文目录) -> UnicodeDecodeError: 'utf-8' codec can't decode byte 0x87
  ```

  而"判断某路径是不是目录"这一步恰好需要列目录 → `./runs`（要判断父目录 `.`）、绝对中文路径**全部失败**
- 对照实验（同一份代码，只换目录/写法）：

  | 工作目录 | 路径写法 | 结果 |
  | --- | --- | --- |
  | 全英文（`C:\...\Temp`、`D:\ML_Project\nlp`） | `./runs` | ✅ |
  | 中文目录 `文本张量表示` | `./runs` | ❌ `. is not a directory` |
  | 中文目录 | `runs`（目录不存在时） | ✅ 侥幸通过 |
  | 中文目录 | `runs`（目录已存在） | ❌ `runs is not a directory` |
  | 中文目录 | 绝对路径 | ❌ `…文本张量表示 is not a directory` |

- **解法：`log_dir` 用全英文绝对路径**，其他代码都不用动：

  ```python
  my_summary = SummaryWriter(log_dir=r'D:\ML_Project\nlp\runs')
  ```

  启动也要一致：`tensorboard --logdir=D:\ML_Project\nlp\runs`
- **治本**：把 `文本张量表示` 改成英文目录名。TF 系（`tf.keras` 存模型、读数据集、SavedModel）以后还会在别处踩这个坑

### 4. TensorBoard 起不来：`No module named 'pkg_resources'`

- 现象：`tensorboard --logdir=...` 直接 Traceback，连 `tensorboard --version` 都失败
- 根因：`setuptools 84.0.0` 已移除 `pkg_resources`，而 `tensorboard 2.19.0` 启动时硬要 import 它
  （TF 2.19 又把 tensorboard 钉在 2.19，升不上去）
- 处理：`pip install "setuptools<81"`，官方警告原文就写着 *pin to Setuptools<81*。当前 **81.0.0**，实测可用
- 验证：`tensorboard --version` → `2.19.0`；实际启动 → HTTP 200

### 5. `add_embedding` 的 metadata 个数必须等于矩阵行数

- 报错：`AssertionError: #labels should equal with #data points`
- 原因：`embd.weight` 是 **24 行**（23 个词 + 1 个预留位），而 `list(tokenizer.word_index.keys())` 只有 **23 个**
- 修法：`metadata=['<PAD>'] + list(tokenizer.word_index.keys())`
- 进阶：**顺序不要赌 `keys()`**，按编号回填更稳（见上面 Day02 第 4 节）

### 6. 相对路径 = 相对"工作目录"，不是相对脚本文件

- 见上面「运行方式」的 ⚠ 说明（`./data/use.txt` 在 PyCharm 默认工作目录下会 `FileNotFoundError`）

### 7. TensorBoard 启动：`--logdir` 别拼成 `--logdis`

- 参数是 **`--logdir`**（dir = directory）。写成 `--logdis=runs` 会直接退出，退出码 2：

  ```
  tensorboard: error: unrecognized arguments: --logdis=runs
  ```

- **命令找不到**（`'tensorboard' 不是内部或外部命令`）：它不在系统 PATH 里（在 `D:\Anaconda3\envs\nlpbase\Scripts\`），
  用完整路径，或先 `conda activate nlpbase`
- **三种启动姿势**（实测通过）：

  ```powershell
  # 1) 绝对英文路径，任何目录都能跑（推荐）
  D:\Anaconda3\envs\nlpbase\Scripts\tensorboard.exe --logdir=D:\ML_Project\nlp\runs
  # 2) 先 cd 到项目根，再用相对路径
  cd D:\ML_Project\nlp ; D:\Anaconda3\envs\nlpbase\Scripts\tensorboard.exe --logdir=runs
  # 3) 局域网 / 手机也能看
  D:\Anaconda3\envs\nlpbase\Scripts\tensorboard.exe --logdir=D:\ML_Project\nlp\runs --host=0.0.0.0 --port=6006
  ```

- `--host 0.0.0.0` 的两个副作用：① 首次会弹 **Windows 防火墙**授权，不放行就只有本机能连；
  ② 它**没有任何登录认证**，等于把事件文件暴露给同网段所有人 —— 只在本机看就别加这一项
- **别在中文目录里启动**，也别把 `--logdir` 指到含中文子目录的父目录，否则又回到第 3 条的 `UnicodeDecodeError`
- 自检"TensorBoard 到底读没读到数据"：访问 `http://localhost:6006/data/plugin/projector/runs`
  → 返回 `["."]` 就说明读到了（run 名显示为 `.`，因为 logdir 直接指向 run 目录本身）
- 它是**常驻服务**，`Ctrl+C` 停止

### 8. pandas 默认读 TSV 会因"引号解析"**静默丢行**（`文本分析/data/train.tsv` 踩到）

`train.tsv` 文件里有 **3000** 行数据，但 `pd.read_csv(sep='\t')` 只读出 **2960** 行，**不报任何错**。

原因：有 28 行句子**本身以英文双引号 `"` 开头**（影评原文里的引号）。pandas 默认 `quoting=QUOTE_MINIMAL`，会把它当成"引号包裹的字段"，一路吞到下一个引号才结束 —— 于是**相邻几行被合并成一行**。

后果（没发现的话，一批统计口径全是错的）：

| 指标 | 错误读法 | 正确读法 |
| --- | --- | --- |
| train 条数 | 2,960 | **3,000** |
| 最长句长 | 3,370 字（合并出来的假句子） | **975 字** |
| 总词数 / 不同词 | 156,952 / 12,162 | **156,963 / 12,160** |

两种修法：

```python
import csv, pandas as pd
df = pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE)   # 推荐
```

```python
# 或者逐行读，自己 split
rows = []
with open(path, encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) == 2 and p[0].strip():
            rows.append((p[0].strip(), int(p[1])))
```

**教训**：读完数据先核对行数（`len(df)` 要能对上文件行数）再做统计。**静默丢行比直接报错危险得多** —— 报错你会去查，丢行你根本不知道。

### 9. 训练统计的"分子分母"必须同一口径（`RNN模型/04_人名国家识别.py` 踩到）

想让曲线平滑，把每 100 步记录的**分母**改成了全局计数器 `seen`：

```python
seen = 0
for epoch in range(epochs):
    total_loss, total_acc_num = 0.0, 0     # ← 分子每轮清零（旧的写法）
    for i, (x, y) in enumerate(dataloader):
        seen += 1
        ...
        if (i + 1) % 100 == 0:
            total_loss_list.append(total_loss / seen)      # 分子:本轮 | 分母:全局 ← 打架
            total_acc_list.append(total_acc_num / seen)
```

**症状**：第 1 轮数字正常，之后越来越离谱——损失降到 0.077（几乎全对）而准确率显示 **17.4%**，而且**低于"全猜 Russian"的 46.9% 基线**。

**根因**：第 1 轮 `seen == i+1`，两者恰好相等所以看不出问题；从第 2 轮起分母被前面几轮撑大，比值必然崩掉。放大倍数正是 `((轮次-1)×每轮样本数 + step) / step`——**轮末那个点永远等于"已完成的轮数"**（第 4 轮轮末 ×4.0000），这是最快的自检方法。

**修法（二选一，不能混）**：

| 想要的效果 | 分子 | 分母 |
| --- | --- | --- |
| 全局平滑曲线（推荐） | **不清零**，一直累加 | `seen` |
| 每轮锯齿曲线 | 每轮清零 | `(i + 1)` |

**这个东西是可以反推救回来的**：存的值 = `本轮累计 ÷ 全局seen`，所以 `本轮累计 = 存的值 × seen`，再用"本轮已处理样本数"当分母就还原了：
`真实值 = 存的值 × (每轮样本数 × 已过轮数 + 本轮已处理数) ÷ 本轮已处理数`。第 1 轮反推结果与原始值完全一致（因为那一轮 `seen` 就等于 `i+1`），这本身就是公式正确的验证。

反推后三模型的真实准确率：GRU 88.2% / LSTM 87.3% / RNN 76.0%，单调上升、无异常。

**教训**：**统计量的分子和分母必须是同一个时间尺度**。改口径时要把"清零的位置"和"分母"一起检查——只改一半，数字会看起来像"模型训崩了"，实际是统计代码错了。

### 10. 解码器里加了 softmax，外面又用 CrossEntropyLoss → 梯度只剩 16%（`英译法/translate.py` 踩到）

`nn.CrossEntropyLoss` **内部自带 `log_softmax`**，所以解码器 `forward` 里**不能**再写 softmax：

```python
output = self.softmax(self.out(output[0]))     # ✗ 外面用 CrossEntropyLoss 时错
output = self.out(output[0])                   # ✓ 直接返回原始 logits
```

实测（10 类的小例子，二者形状都合法，所以不会报错）：

| 写法 | loss | **梯度模长** |
| --- | --- | --- |
| 先 softmax 再 `CrossEntropyLoss` | 2.252067 | **0.059793** |
| 直接 `CrossEntropyLoss(logits)` ✓ | 2.060659 | **0.374174** |

**梯度只剩 16%（小了 84%）。** 原因是 `log_softmax(概率) ≠ log(概率)`：CE 会对已经归一化的概率**再做一次** `log_softmax`，把分布抹平：

```
log(概率)         = [-2.525, -1.913, -3.102]
log_softmax(概率) = [-2.325, -2.257, -2.360]   ← 实际算的是这个
```

**三种合法组合**：无 softmax + `CrossEntropyLoss`（推荐）｜ `nn.LogSoftmax` + `nn.NLLLoss` ｜ `nn.Softmax` + **没有**匹配的损失。

**顺带**：`nn.Softmax()` 不写 `dim` 会触发 `UserWarning: Implicit dimension choice for softmax has been deprecated`。
二维张量上隐式选 `dim=1` 恰好是词表维所以"侥幸对"，但张量变成 3 维就会在**错误的维度**归一化。**永远写 `dim=-1`。**

**教训**：这类错误 **loss 曲线照样在降**，只是慢——不看梯度就查不出来。**"不报错"不等于"对"。**

### 11. 训练时不喂 SOS，SOS 的 embedding 永远学不到 → 推理第一步就崩（`英译法/translate.py` 踩到）

解码器的输入必须**右移一位且以 SOS 开头**：

```
法文原句:              je    suis   etudiant   .
目标 y（数据集返回）:  je    suis   etudiant   .   [EOS]
解码器输入:        [SOS]  je    suis   etudiant   .          ← [SOS] + y[:-1]
```

**不塞 SOS 的两条错路**：

| 错法 | 后果 |
| --- | --- |
| 输入 = 目标（不右移） | 第 0 步"输入 w1、要求输出 w1" = **抄输入**，模型学个恒等映射就能把 loss 压低，永远学不到翻译 |
| 右移了但无 SOS | 第 1 个词**永远不会被预测**；且推理时第一步喂 SOS，模型训练时从没见过它 |

**最硬的证据**（实测 embedding 梯度）：

```
输入含 SOS   [0, 3, 4, 5]  → SOS 那一行梯度【非零】✓ 会被更新
输入不含 SOS [3, 4, 5, 6]  → SOS 那一行梯度【全 0】✗ 永远是随机初始化值
                             有梯度的行号只有 [3, 4, 5, 6]
```

推理时第一步喂的就是 SOS，而它是个**从没被训练过的随机向量** → 输出乱码。

**正确写法**（两种等价）：

```python
# A 循环 + 逐步喂真值（贴近课程风格）
dec_input = torch.tensor([[SOS_TOKEN]], device=device)
for j in range(y.shape[1]):                    # ★ 用 j，别覆盖外层的 i
    logits, hidden = decoder(dec_input, hidden)
    dec_input = y[0][j].view(1, -1)            # ★ 喂【真值】= teacher forcing

# B 一次喂整句（decoder 支持 seq 输入时，更快）★ 推荐
dec_input = torch.cat([torch.tensor([SOS_TOKEN], device=device), y[0][:-1]]).unsqueeze(0)
logits, _ = decoder(dec_input, enc_hidden)
loss = F.cross_entropy(logits.view(-1, FRENCH_VOCAB), y.view(-1))
```

**注意课程的原版代码喂的是 `y[0][i]`（没有 SOS）** —— 抄课程时要自己补上这一步。

**教训**：**"训练时要喂的输入"和"推理时要喂的输入"必须一致**。训练时没出现过的符号，推理时给它就是垃圾进垃圾出。

### 12. Adam 配 `lr=0.01` 会发散：loss 越训越大（`英译法/translate.py` 踩到）

现象：跑了几百步后，**loss 不是下降而是往上走** ——

```
Epoch [1/10], Step [1000/63594], print_avg_loss: 13.4108
Epoch [1/10], Step [2000/63594], print_avg_loss: 19.3603
Epoch [1/10], Step [3000/63594], print_avg_loss: 20.2607   ← 越来越大
```

**复现与定位**（同一份代码、同一个初始化，各跑 1000 步，loss 统一按"每 token 平均"报告）：

| 配置 | 第100步 | 第500步 | 第1000步 | 结论 |
| --- | --- | --- | --- | --- |
| **Adam lr=0.01** | 8.87 | 15.85 | **22.22** | ✗ **发散** |
| Adam lr=0.001 | 4.85 | 3.33 | **3.06** | ✓ 正常 |
| Adam lr=0.0001 | 更慢 | — | — | ✓ 稳，但要多训几轮 |

**原因**：`lr=0.01` 对 **Adam** 来说大了约 10 倍（Adam 常用 `1e-3`）。Adam 有自适应步长，参数被一点点推离有效区域 → loss 越训越大。

**注意：官方教程用的是 `lr=0.01`，但配的是 `SGD`** —— 换 Adam 就必须改小。

**顺带一个易误判的点**：`loss` 用 sum 还是 mean 去 `backward`？

- **实测 Adam 下两者几乎没差别**（4.85→3.06 vs 4.86→3.10）：Adam 的更新是 `m/(√v+ε)`，梯度整体乘常数时 `m` 和 `√v` 同比例变，**商不变**
- **但 SGD 下差 ~7 倍**（梯度范数 59.33 vs 7.77）→ 等于 lr 放大 7 倍
- 所以"先取平均"不是数学上更正确（序列 NLL 本来就是求和），而是**工程上更稳 + 上 batch 后天然按 token 平均**

**唯一必须改的是口径一致**：`return loss.item()/L` 却用 `loss`（sum）去 backward，会让"你看到的 loss"和"实际用的梯度"不是一个东西 —— 同类问题见踩坑记录第 9 条。

**教训**：
1. **`loss` 不降反升 = 学习率太大**（或梯度爆炸），第一反应就查 lr
2. **抄教程的超参数要看配套的优化器**（SGD 的 0.01 ≠ Adam 的 0.01）
3. 判断"训练是不是真的在学"：随机初始化的 loss 应该是 `ln(词表大小)`（本项目 = `ln(4345) ≈ 8.38`）

### 13. `pack` 和 `mask` 不能互相替代 —— 它们修的是两台不同的机器（`英译法/translate.py` 改造时踩到）

**以为"做了 pack 就不用 mask"（或反过来）是错的。** 两者作用在不同位置：

| | 毛病 | 谁修 | 作用点 |
| --- | --- | --- | --- |
| 编码器 **GRU** | 多读几步 PAD → `hidden` 永久变脏 | **`pack`** | 读之前拦 |
| 解码器 **softmax** | 多几个候选 → 真词权重被稀释 | **`mask`** | 归一化前拦 |

**★ 决定性实验 A：mask 替代不了 pack**

同一句话，只改"补到多长"，看 `hidden`：

| 做法 | 与"不补"的差 |
| --- | --- |
| 补到 10，不 pack | **0.6442** |
| ↑ 事后把输出 mask 成 0 | **0.6442** ← ★ 一点没变 |
| ↑ 改用 pack | **0.00000** ← ✓ |

> **`mask` 作用于"输出张量"，而 `hidden` 是 GRU 内部逐时间步累加出来的状态。**
> **一旦读了 PAD，那一步已经走过了 —— 事后 mask 输出救不回状态。**

**★ 决定性实验 B：pack 替代不了 mask**

`pad_packed_sequence` 的输出**又变回带 PAD 的矩形**（因为下游 `bmm` 要求规整矩阵）：

```
pack 之后 .data             (11, 512)     ← 只有真实词（7+4）
pad_packed_sequence 之后     (2, 7, 512)   ← ★ PAD 回来了，填 0
```

而行1（真实 4 词）的注意力：

```
不 mask: [0.0901, 0.1133, 0.1628, 0.3558, 0.0927, 0.0927, 0.0927]   真词和 0.7220
带 mask: [0.1249, 0.1570, 0.2254, 0.4927, 0.0,    0.0,    0.0   ]   真词和 1.0000
```

**3 个 PAD 合计抢走 27.8% 的注意力。** 它们是 0 向量，但 **softmax 只看"有几个候选"，不管值是 0**。

**★ 为什么不能"一直保持 packed"**：`PackedSequence` 不是 Tensor（`'PackedSequence' object has no attribute 'transpose'`），没有 batch 维、没有统一 L 维，**做不了 `bmm`**；PyTorch 也没有"变长分组 softmax"的原语。

**顺带一个致命细节**：`PAD` 编号**不能用 0**。因为 `nn.Embedding(..., padding_idx=0)` 会把第 0 行冻成 0 且永远不给梯度，而本项目 `SOS=0` → **SOS 会被一起冻死**（实测梯度 `[0,0,0,0]`），解码器第一步就喂进全 0 垃圾。

### 14. `masked_fill` 的两个手工细节都不能省，且**都不报错**（`英译法/translate.py` 踩到）

```python
score = score.masked_fill(~enc_mask.unsqueeze(1), float('-inf'))
```

**细节 1：`~`（取反）**

`masked_fill` 填的是 **`True` 的位置**，而 mask 的约定是 `True`=真词，两者方向相反：

```
score [1.0, 2.0, 3.0]   mask [True, True, False]
masked_fill(mask,  -inf) → [-inf, -inf, 3.0]    ← ✗ 把真词填死了
masked_fill(~mask, -inf) → [1.0,  2.0,  -inf]   ← ✓
```

**细节 2：`unsqueeze(1)`**

```python
score    (B, 1, L)
enc_mask (B, L)          ← 少了中间那一维
```

**漏了它会静默广播**，形状从 `(B,1,L)` 变成 `(B,B,L)`，**报错要等到 3 步之后的 `torch.cat`**：

```
① masked_fill 后  (3,1,8) → ★ (3,3,8)
② softmax 后      (3,3,8)
③ bmm → (3,3,512)
④ cat(embedded(3,1,256), context(3,3,512))
   → ✗ Sizes of tensors must match except in dimension 2. Expected size 1 but got size 3
```

**还有两个容易忽略的点**：

- **`-inf` 而不是 `0`**：`softmax(0) = e⁰ = 1`，填 0 时 PAD 仍分到 **5.2%** 权重；填 `-inf` 才精确为 0
- **必须在 softmax 之前**：先 softmax 再置 0，权重和变成 **0.9618**（归一化被破坏，因为分母已经把 PAD 算进去了）

**边界**：整行都 mask → `softmax([-inf,-inf,-inf]) = [nan,nan,nan]`，NaN 会顺着梯度污染整个模型。

### 15. `nn.Module` 里**方法名和属性名撞了**会无限递归，且 `get_submodule` 直接崩（`transformer架构/` 踩到）

课程代码里同时写了 `self.encoder = encoder`（模块）和 `def encoder(self, ...)`（方法）。

**关键机制**：`nn.Module.__setattr__` 会把模块存进 **`_modules`**，**不进 `__dict__`**：

```
m.__dict__ 里有 'encoder' 吗？   False      ← 关键
m._modules 里有 'encoder' 吗？   True
type(m.encoder).__name__       = method     ← 拿到的是【方法】不是模块
```

属性查找顺序是「实例 `__dict__` → 类字典」，`__dict__` 里没有 → **找到类上的方法 → 方法胜出**。

```python
def encoder(self, source_x, source_mask):
    return self.encoder(embedded, source_mask)   # ← 调用的是【自己】
# ✗ RecursionError: maximum recursion depth exceeded
```

**而且 `decode` 却没事** —— 因为属性叫 `decoder`（多一个 `r`），和方法名 `decode` 错开了。

**连带破坏的 API**：

| 检查项 | 结果 |
| --- | --- |
| `state_dict()` / `named_modules()` / `parameters()` | ✓ 正常（内部直接读 `_modules`，会给你"能跑"的错觉） |
| `model.encoder` | ✗ 拿到的是方法 |
| `model.get_submodule('encoder')` | ✗ `AttributeError: encoder is not an nn.Module` |

**修法**：方法改名 `encode`（属性 `self.encoder` ↔ 方法 `encode`，错开一个字母）。或者干脆不写包装，`forward` 里直接调 `self.encoder(...)`。

> **规范**：`self.encoder` / `def encode`、`self.decoder` / `def decode`、`self.norm` / `def normalize` —— **别同名**。
> 自检：`print(type(model.encoder))` 显示 `method` 就是撞了。

### 16. 位置编码没接上，模型**完全分不出词序**，但照样能训（`transformer架构/dm09_transformer.py` 踩到）

`PositionalEncoding` 写好了，却忘了放进 `src_embed`：

```python
src_embed = nn.Sequential(Embedding(vocab, d_model))                       # ❌ 漏了
src_embed = nn.Sequential(Embedding(vocab, d_model), PositionalEncoding(d_model))  # ✅
```

**为什么致命**：Transformer 的所有子层都**位置对称**（自注意力对排列不敏感、FFN 逐位置同参数、LayerNorm 逐位置独立、残差逐元素），所以整个编码器满足 `enc(perm(x)) = perm(enc(x))`。

**实测（同一批词打乱顺序，比较还原后的输出）**：

| `src_embed` 组成 | 最大差 | 能区分词序 |
| --- | --- | --- |
| 只有 `Embedding` | **0.000001** | ❌ 数学上完全相等 |
| `Embedding + PositionalEncoding` | **0.522509** | ✓ |

> **模型会把"我爱你"和"你爱我"当成同一个东西，但 loss 照样在降、训练照样"成功"。**

**常备自检**：

```python
ids = torch.tensor([[5, 7, 9, 11]]);  perm = torch.tensor([2, 0, 3, 1])
h1 = model.encoder(model.src_embed(ids))
h2 = model.encoder(model.src_embed(ids[:, perm]))
assert (h2[0] - h1[0][perm]).abs().max() > 1e-3, '位置编码没生效！'
```

### 17. `mask` 传 4 维会被 `unsqueeze(1)` 加成 5 维，报错却在 `permute` 里（`dm02_encoder_element.py` 踩到）

```python
if mask is not None:
    mask = mask.unsqueeze(1)      # 无条件插一维
```

| 传进来的 `mask` | unsqueeze 后 | 结果 |
| --- | --- | --- |
| `(B, L, L)` 3 维 | `(B, 1, L, L)` | ✓ |
| `(B, 1, L, L)` 4 维 | `(B, 1, 1, L, L)` | ✗ **变 5 维** |
| `(L, L)` 2 维 | `(L, 1, L)` | ✗ 位置插错，广播错位 |

**4 维的崩法**：`masked_fill` 时 broadcasts 出 `(1,2,8,4,4)` 的 5 维张量 → 一路传到 `output.permute(0,2,1,3)` 才炸：

```
RuntimeError: permute(sparse_coo): input.dim() = 5 is not equal to len(dims) = 4
```

**报错信息完全没提 mask**，很难定位。**修法**：

```python
if mask is not None and mask.dim() == 3:
    mask = mask.unsqueeze(1)
```

**顺带记住 `src_mask` 唯一通用的形状是 `(B, 1, L_src)`** —— 它要同时喂编码器自注意力 `(B,h,L_src,L_src)` 和交叉注意力 `(B,h,L_tgt,L_src)`，只有这一种能两边广播通过。

## 说明

- 课程资料版权归黑马程序员所有，本仓库仅存放个人练习代码与笔记
- **文档更新约定**：每次学习结束后，由 AI 助手同步刷新本文件与 [`课程进度.md`](课程进度.md)，
  改完**直接提交**，提交信息固定为 **「更新文档」**；`git push` 由我自己执行
- 上一个项目的踩坑经验（OpenMP 冲突、`batch_size` 对 GPU 利用率的影响、`CrossEntropyLoss` 的类别维要求等）见 `D:\ML_Project\Deep Learn\README.md`，NLP 阶段还会再遇到

## 参考

- [黑马程序员《AI 大模型 NLP 自然语言处理保姆级教程》](https://www.bilibili.com/video/BV14mdfBDE4Q/)
- [jieba 中文分词（GitHub）](https://github.com/fxsjy/jieba)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers/index)
