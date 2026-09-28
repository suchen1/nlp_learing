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
| Day02 | 词向量与文本分析 | P18–P35 | ⬜ 未开始 |
| Day03 | 文本规范 + RNN / LSTM / GRU | P36–P54 | ⬜ 未开始 |
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

## 环境

| 项 | 版本 / 说明 |
| --- | --- |
| 操作系统 | Windows |
| 项目环境 | **`nlpbase`**（conda）｜`D:\Anaconda3\envs\nlpbase\python.exe` |
| Python | 3.10.21（conda-forge） |
| TensorFlow | **2.19.0**（Day01 的 one-hot 用的是 `tensorflow.keras` 的 Tokenizer） |
| NumPy | 2.1.3 |
| jieba | 0.42.1 |
| joblib | 1.5.1（保存 / 加载 tokenizer） |
| IDE | PyCharm 2026.1（`.idea/` 随项目） |

> **为什么单独开一个环境**：上一个深度学习项目用的 `base`（`C:\ProgramData\anaconda3`，Python 3.14.6 + torch 2.14.0+cu130）
> Python 版本太新，而 NLP 这套要装 TensorFlow / fasttext / transformers，这些库在 **3.10** 上踩坑最少。
> `base` 那套没动，随时能切回去；`learn` 环境里只有 jieba，暂时用不上。
>
> PyCharm 切换方式：Settings → Project: nlp → Python Interpreter → Add Interpreter → Conda Environment → **Use existing** → 选 `nlpbase`。

### 依赖现状（2026 年排查，基于 `nlpbase`）

**已装**：`tensorflow 2.19.0`、`numpy 2.1.3`、`jieba 0.42.1`、`joblib 1.5.1`

课程接下来会用到、目前**还没装**：

| 依赖 | 从哪里开始需要 | 备注 |
| --- | --- | --- |
| `torch` + `torchvision` | **Day03（P36 起）就要用** | 必须装 GPU 版，见下面「GPU 版 PyTorch」 |
| `pandas` | Day02 文本分析（读数据、看标签分布） | — |
| `matplotlib` | Day02 句子长度分布、词云 | 记得配 `SimHei` 中文字体 |
| `scikit-learn` | 后面的评估、切分数据 | — |
| `fasttext` | Day02 词向量、Day10 Fasttext 优化 | Windows 上常要现场编译，装不上再想办法 |
| `wordcloud` | Day02 高频形容词词云 | — |
| `transformers` `datasets` | Day10–Day12 预训练模型与迁移学习 | 体积大，用到再装 |
| `tensorboard` | 词向量可视化 | — |
| `opencc` | 繁转简（Day01 提过） | 可选 |

按课程进度**用到再装**，不要一上来全装 —— 环境干净，出问题好定位。

#### GPU 版 PyTorch（Day03 之前要装好）

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
├── 文本张量表示/               Day01 P12–P16  one-hot 编码
│   ├── 01_one-hot.py                          用 tensorflow.keras 的 Tokenizer 取 one-hot，joblib 存读
│   └── model/tokenizer.pkl                    存下来的词表（word_index）
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
| P17 | word2vec CBOW 原理 | 纯理论，无代码（要点见上面「知识点笔记」第 4 节） |

## 运行方式

```powershell
# 项目环境是 nlpbase，直接用它的解释器
D:\Anaconda3\envs\nlpbase\python.exe jieba\01_jieba使用.py
```

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

（暂无）

## 说明

- 课程资料版权归黑马程序员所有，本仓库仅存放个人练习代码与笔记
- 上一个项目的踩坑经验（OpenMP 冲突、`batch_size` 对 GPU 利用率的影响、`CrossEntropyLoss` 的类别维要求等）见 `D:\ML_Project\Deep Learn\README.md`，NLP 阶段还会再遇到

## 参考

- [黑马程序员《AI 大模型 NLP 自然语言处理保姆级教程》](https://www.bilibili.com/video/BV14mdfBDE4Q/)
- [jieba 中文分词（GitHub）](https://github.com/fxsjy/jieba)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers/index)
