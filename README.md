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
| 导学 | 课程导学 | P1 | ⬜ 未开始 |
| Day01 | 文本预处理：分词 / 词性标注 / NER / one-hot | P2–P17 | 🟡 进行中 |
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

## 环境

| 项 | 版本 / 说明 |
| --- | --- |
| 操作系统 | Windows |
| Python | 3.14.6（Anaconda，`C:\ProgramData\anaconda3`） |
| PyTorch | **2.14.0+cu130**（CUDA 可用） |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| Matplotlib | 3.11.0 |
| jieba | 0.42.1 |
| IDE | PyCharm（`.idea/` 随项目） |

> 与上一个深度学习项目是**同一套环境**，GPU 相关配置、`KMP_DUPLICATE_LIB_OK` 之类的坑可以照旧复用。

### 依赖现状（2026 年排查）

已装：`torch` `numpy` `pandas` `matplotlib` `scikit-learn` `jieba`

课程后面会用到、目前**还没装**：

| 依赖 | 用在哪里 | 备注 |
| --- | --- | --- |
| `fasttext` | Day02 词向量、Day10 Fasttext 优化 | Python 3.14 比较新，如果 pip 没有现成 wheel 可能会要求本地编译，到时再折腾 |
| `wordcloud` | Day02 高频形容词词云 | — |
| `transformers` | Day10–Day12 预训练模型与迁移学习 | 顺带会需要 `datasets`、`torch` 之外的 `safetensors` 等 |
| `datasets` | Day10–Day12 | — |
| `tensorboard` | 词向量可视化 | — |
| `opencc` | 繁体转简体（Day01 分词会聊到） | 可选 |

按课程进度**用到再装**，不要一上来全装 —— 环境干净，出问题好定位。

## 目录约定

课程按「天」推进，代码建议按模块分目录（名字自己顺手就行）：

```
nlp/
├── 文本预处理/      Day01  分词、词性标注、NER、one-hot
├── 词向量/          Day02  SkipGram、FastText、WordEmbedding、文本分析
├── RNN/             Day03  RNN / LSTM / GRU 的原理与代码
├── 人名分类/        Day03–Day04  三模型对比实战
├── 注意力机制/      Day05  QKV、Seq2Seq、注意力代码实现
├── Seq2Seq英译法/   Day06–Day07  英译法案例
├── Transformer/     Day07–Day09  逐层搭建：位置编码→多头注意力→编码器/解码器→拼装
├── Fasttext/        Day10  Fasttext 训练与优化
├── 预训练模型/      Day10–Day12  Transformers 库、pipeline、BERT 迁移学习
├── data/            自己的语料与数据集
├── README.md
└── 课程进度.md      189 P 完整课程地图与进度清单
```

## 运行方式

命令行直接运行单个脚本：

```bash
C:\ProgramData\anaconda3\python.exe 文本预处理\01_xxx.py
```

或在 PyCharm 中打开对应文件，按 `Shift+F10` 运行。

> 在 Windows 上如果 `python` 命令被 Microsoft Store 的应用别名占用，用上面的完整路径。
>
> 脚本里读写数据建议用 `Path(__file__).parent` 拼路径，而不是 `./data/xxx` ——
> 这样在 PyCharm 里直接点运行、或者在任意目录下用命令行运行，都能找到文件。

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
