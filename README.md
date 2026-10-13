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
| Day03 | 文本规范 + RNN / LSTM / GRU | P36–P54 | 🟡 进行中（已到 P49 GRU 代码） |
| Day04 | 人名分类案例实战 | P55–P65 | ⬜ 未开始 |
| Day05 | 注意力机制 | P66–P79 | ⬜ 未开始 |
| Day06 | Seq2Seq 英译法案例（上） | P80–P95 | ⬜ 未开始 |
| Day07 | 英译法案例（下）+ Transformer 登场 | P96–P109 | ⬜ 未开始 |
| Day08 | Transformer 逐层实现（上） | P110–P125 | ⬜ 未开始 |
| Day09 | Transformer 逐层实现（下） | P126–P143 | ⬜ 未开始 |
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

### Day02 词向量与 WordEmbedding（P18–P25）

> 当前进度：Day02 进行中，已到 **P24 WordEmbedding 获取词向量**。下一个是 **P25 WordEmbedding 查看词向量的相似性**。

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

### Day03 文本规范 + RNN / LSTM（P36–P54）

> 当前进度：已到 **P49 GRU 模型_代码演示**，下一个是 **P50 RNN人名分类案例_需求介绍**。
>
> **Day03 的详细笔记单独成篇**（内容较长，不往 README 里塞）：
> - 📄 [`RNN模型/LSTM反向传播.md`](RNN模型/LSTM反向传播.md) — LSTM 到底有几组权重、四个门的梯度怎么更新、为什么 `c_t` 是"梯度高速公路"（全部带实测数字，文末附可复制的复现代码）

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
| IDE | PyCharm 2026.1（`.idea/` 随项目） |

> **为什么单独开一个环境**：上一个深度学习项目用的 `base`（`C:\ProgramData\anaconda3`，Python 3.14.6 + torch 2.14.0+cu130）
> Python 版本太新，而 NLP 这套要装 TensorFlow / fasttext / transformers，这些库在 **3.10** 上踩坑最少。
> `base` 那套没动，随时能切回去；`learn` 环境里只有 jieba，暂时用不上。
>
> PyCharm 切换方式：Settings → Project: nlp → Python Interpreter → Add Interpreter → Conda Environment → **Use existing** → 选 `nlpbase`。

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
├── RNN模型/                    Day03 P41 RNN ｜ P44–P49 LSTM / GRU
│   ├── 01_RNN.py                              P41 nn.RNN 的形状演示（input/output/hn）
│   ├── 02_LSTM.py                             P47 nn.LSTM + h0/c0，output/h1/c1 形状
│   ├── 03_GRU.py                              P49 nn.GRU（注意：只有 h，没有 c）
│   └── LSTM反向传播.md                         Day03 LSTM 反向传播笔记（课程没展开的部分，文末附复现代码）
├── runs/                      TensorBoard 事件文件（生成物，已 gitignore）
├── .gitignore                 挡 4 个大文件 + runs/
├── README.md
└── 课程进度.md                 189 P 完整课程地图与进度清单
```

后续目录（学到再建）：`词向量/` Day02 · `RNN/` Day03 · `人名分类/` Day03–04 · `注意力机制/` Day05 · `Seq2Seq英译法/` Day06–07 · `Transformer/` Day07–09 · `Fasttext/` Day10 · `预训练模型/` Day10–12

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

或在 PyCharm 里打开对应文件按 `Shift+F10`。

> ⚠ **工作目录很重要**：现有脚本里用的是相对路径（`./data/use.txt`、`./model/tokenizer.pkl`），
> 它们相对于**运行时的工作目录**，不是相对于脚本文件本身。
> PyCharm 默认把 Working directory 设成**项目根目录**（`D:\ML_Project\nlp`），
> 那样 `./data/use.txt` 会被解释成 `D:\ML_Project\nlp\data\use.txt` → `FileNotFoundError`。
> 两个办法，任选一个：
> 1. 改运行配置：Run → Edit Configurations → **Working directory** 填 `$FileDir$`（脚本所在目录），一劳永逸；
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
