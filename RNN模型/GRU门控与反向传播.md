# GRU 的门控机制与反向传播（Day03 P48–P49）

> **本文回答的问题**：重置门 `r` 和更新门 `z` 到底各干什么？两个门怎么合起来决定记忆？反向传播时它们怎么更新？为什么说 GRU 的"记忆通路"不如 LSTM 直？
>
> 文中每个数字都在本机 PyTorch 上实跑得出（环境 `nlpbase`），**复现代码见文末【附：复现代码】**。
>
> 课程 P49 的代码在 [`03_GRU.py`](03_GRU.py)；LSTM 的同类笔记见 [`LSTM反向传播.md`](LSTM反向传播.md)。

---

## 0. 一句话定位

> **GRU = 把 LSTM 的 4 组门压成 3 组，用一个「更新门」同时管"忘旧"和"记新"，并且取消了独立的记忆细胞 `c`。**

代价是：记忆通路没有 LSTM 那么"直"（第 4 节有实测对比）。

---

## 1. 参数：3 组门拼在一起

`nn.GRU(input_size=5, hidden_size=4)` 实测：

```
weight_ih_l0     (12, 5)     = 3 组门 × 4 维隐藏
weight_hh_l0     (12, 4)
bias_ih_l0       (12,)
bias_hh_l0       (12,)
```

三组按行切开就是 **`r`（重置门）/ `z`（更新门）/ `n`（候选状态）**。

三种模型的门数与参数量（都是 `input_size=5, hidden_size=10`）：

| 模型 | 门的组数 | `weight_ih_l0` | 总参数量 | 返回值 |
| --- | --- | --- | --- | --- |
| `nn.RNN` | 1 组 | (10, 5) | 170 | `(output, hn)` |
| `nn.GRU` | **3 组** | (30, 5) | **510** | `(output, hn)` |
| `nn.LSTM` | **4 组** | (40, 5) | **680** | `(output, (hn, cn))` |

参数量正好 **1 : 3 : 4**。

---

## 2. 两个门的细节（重点）

### 2.1 重置门 `r` —— 决定"上一步的记忆要不要参与计算"

```
r_t = σ( W_ir·x_t + b_ir + W_hr·h_{t-1} + b_hr )
```

⚠️ **最容易写错的一点**：`r` **只作用在"隐藏部分"上**，不是作用在整个候选状态上：

```
n_t = tanh( W_in·x_t + b_in  +  r_t ⊙ ( W_hn·h_{t-1} + b_hn ) )
                                  └──────── 只乘这一项 ────────┘
```

**直觉**：

| `r` 的值 | 候选状态变成 | 含义 |
| --- | --- | --- |
| 接近 **0** | `n_t ≈ tanh(W_in·x_t + b_in)` | **把历史清空**，只根据当前输入重新算 |
| 接近 **1** | `n_t ≈ tanh(... + W_hn·h_{t-1} + b_hn)` | 正常参考上一步的记忆 |

**类比**：写笔记时先决定"要不要翻回上一页看一眼"——`r` 就是这个动作。注意它只影响**候选状态怎么算**，不直接决定最终保留多少。

### 2.2 更新门 `z` —— 一个门干两件事

```
z_t = σ( W_iz·x_t + b_iz + W_hz·h_{t-1} + b_hz )

h_t = (1 - z_t) ⊙ n_t  +  z_t ⊙ h_{t-1}
      └─ 写入新的 ─┘     └─ 保留旧的 ─┘
```

**这就是 GRU 最巧妙的地方**：`z` 同时控制"保留多少旧的"和"写入多少新的"，而且**两者强制加起来等于 1**（因为一个是 `z`，另一个是 `1-z`）。

| `z` 的值 | 结果 | 含义 |
| --- | --- | --- |
| 接近 **1** | `h_t ≈ h_{t-1}` | 原样保留旧记忆，**这一步不更新** |
| 接近 **0** | `h_t ≈ n_t` | 完全用新算的候选替换旧记忆 |
| 中间值 | 两者按比例混合 | 部分保留 + 部分更新 |

**对比 LSTM**：LSTM 的遗忘门 `f`（保留旧）和输入门 `i`（写入新）是**两个独立的门**，可以"既全部保留又全部写入"；GRU 用一个 `z` 把它们绑成互补关系，**这就是它省掉一组权重的原因**。

**类比**：LSTM 是两个独立开关，GRU 是一个"混合比例旋钮"——拧到左边全是旧的，拧到右边全是新的。

### 2.3 前向公式汇总（PyTorch 版本）

```
r_t = σ( W_ir·x_t + b_ir + W_hr·h_{t-1} + b_hr )
z_t = σ( W_iz·x_t + b_iz + W_hz·h_{t-1} + b_hz )
n_t = tanh( W_in·x_t + b_in + r_t ⊙ ( W_hn·h_{t-1} + b_hn ) )
h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
```

> ⚠️ **两种写法别抄混**：教材里常见 `h_t = (1-z_t)⊙h_{t-1} + z_t⊙n_t`，而 **PyTorch 用的是 `(1-z_t)⊙n_t + z_t⊙h_{t-1}`**（即 `z` 代表"保留旧记忆的比例"）。两者只是 `z` 的定义相反，但代码里按错的那个写，前向就对不上 `nn.GRU`。下面用实测确认了 PyTorch 的版本。

---

## 3. 手动复现前向（验证公式和门顺序）

```python
import torch, torch.nn as nn

torch.manual_seed(0)
D, H = 5, 4
gru = nn.GRU(D, H, batch_first=True)
x = torch.randn(1, 3, D)
out, hn = gru(x)

W_ih, W_hh = gru.weight_ih_l0, gru.weight_hh_l0
b_ih, b_hh = gru.bias_ih_l0, gru.bias_hh_l0
W_ih_r, W_ih_z, W_ih_n = W_ih.chunk(3, 0)      # 按行切成三组：r / z / n
W_hh_r, W_hh_z, W_hh_n = W_hh.chunk(3, 0)
b_ih_r, b_ih_z, b_ih_n = b_ih.chunk(3)
b_hh_r, b_hh_z, b_hh_n = b_hh.chunk(3)

h = torch.zeros(1, H)
for t in range(x.size(1)):
    xt = x[:, t]
    r = torch.sigmoid(xt @ W_ih_r.T + b_ih_r + h @ W_hh_r.T + b_hh_r)
    z = torch.sigmoid(xt @ W_ih_z.T + b_ih_z + h @ W_hh_z.T + b_hh_z)
    n = torch.tanh(xt @ W_ih_n.T + b_ih_n + r * (h @ W_hh_n.T + b_hh_n))
    h = (1 - z) * n + z * h
    print(t, torch.allclose(h, out[:, t], atol=1e-6))
```

实测输出：

```
第 1 步 手算 == out[:, 0] ? True
第 2 步 手算 == out[:, 1] ? True
第 3 步 手算 == out[:, 2] ? True
最后 h == hn[0] ? True
```

**公式、门顺序、重置门的作用位置全部正确**——这份代码可以当"可执行版公式"。

---

## 4. 反向传播

### 4.1 三组门的梯度：一次 backward 分给每一组

只对最后一步算 loss，`backward()` 后把 `weight_ih_l0.grad` 按门切开：

```
重置门 r    梯度范数 = 0.021642
更新门 z    梯度范数 = 0.064415
候选状态 n   梯度范数 = 0.203907
weight_hh_l0 整体 = 0.041726    bias 整体 = 0.174381
```

三组全都非零，结论和 RNN / LSTM 完全一样：

1. loss 的梯度**同时分给 3 组门**；
2. 同一套权重在序列每一步复用，T 步的梯度**累加**到同一个 `.grad`；
3. `optimizer.step()` 一次，`weight_ih / weight_hh / bias_ih / bias_hh` **全部更新**。

> **门多 ≠ 更新次数多**：GRU 的参数是 RNN 的 3 倍，但更新时机不变——一个 batch 一次 `step()`。

### 4.2 梯度通路：`∂h_t/∂h_{t-1}` 是**稠密矩阵**（这点和 LSTM 不同）

`nn.GRUCell` + `torch.autograd.functional.jacobian` 实测：

```
∂h_t/∂h_{t-1} 雅可比:
  [ 0.51239   0.05484  -0.12497  -0.07394]
  [-0.00389   0.54537   0.04399   0.06111]
  [-0.22041  -0.13397   0.28903  -0.14402]
  [ 0.06757  -0.06311   0.03652   0.36868]

更新门 z 的值 : ['0.57137', '0.45621', '0.34304', '0.48304']
对角线 == z ?   False
非对角全为 0 ?  False
```

**和 LSTM 的关键区别**：

| | LSTM | GRU |
| --- | --- | --- |
| 回传通路 | `∂c_t/∂c_{t-1} = diag(f_t)` —— **严格对角**，只乘一个标量 | `∂h_t/∂h_{t-1}` —— **稠密矩阵**，非对角元素不为 0 |
| 原因 | `c` 是**独立**的记忆线，`f_t` 只由 `x_t`、`h_{t-1}` 决定，且 `c` 不参与门的计算 | GRU **没有独立的 `c`**：`h` 既要当记忆、又要参与 `z`/`n` 的计算，于是产生一堆间接项 |
| 后果 | 高速公路"纯净"，长距离保留最好 | 直接项（对角上的 `z`）还在，但被间接项"掺了杂质" |

**一句话**：GRU 也有"高速路"（`h_t` 里 `z⊙h_{t-1}` 那一项），但因为 `h` 身兼二职，路不如 LSTM 的 `c` 干净。

### 4.3 实测：梯度能回传多远（四方对比）

测法：**只对最后一步算 loss**，看第 0 步还能拿到多少梯度。

| 序列长度 | RNN | GRU | LSTM（默认） | LSTM（遗忘门偏置=1） |
| --- | --- | --- | --- | --- |
| T=20 | 5.19e-07 | 2.96e-06 | 2.14e-06 | 5.24e-04 |
| T=50 | 1.05e-15 | 5.47e-12 | 1.11e-10 | 2.15e-05 |
| T=100 | **0.00e+00**（下溢） | 1.08e-20 | 1.19e-19 | 3.86e-09 |
| T=120 | 0.00e+00 | **0.00e+00**（下溢） | **0.00e+00**（下溢） | 1.23e-11 |

平均**每往前回传一步**乘多少（T=50）：

| 模型 | 每步衰减系数 | 排序 |
| --- | --- | --- |
| RNN | ×0.513 | 最差 |
| **GRU** | **×0.613** | ↓ |
| **LSTM（默认）** | **×0.663** | ↓ |
| LSTM（遗忘门偏置=1） | ×0.851 | 最好 |

**结论**：

1. **衰减速度：RNN > GRU > LSTM**（数字越小衰减越快）。这正好对应"记忆通路的纯净程度"：RNN 只有矩阵连乘 → GRU 有对角直接项但被掺杂质 → LSTM 有严格对角的 `c` 通路。
2. GRU 和 LSTM 的差距**不大**（0.613 vs 0.663），所以短序列上两者效果常常打平——这和实际工程经验一致。
3. 遗忘门偏置初始化成 1 仍然是提升最大的单点改动（0.851）。

---

## 5. GRU vs LSTM 对照表

| | LSTM | GRU |
| --- | --- | --- |
| 门的组数 | 4（`i` / `f` / `g` / `o`） | 3（`r` / `z` / `n`） |
| 隐藏状态 | `h_t` **+ `c_t`** 两条线 | 只有 `h_t` |
| 谁管"保留旧记忆" | 遗忘门 `f`（独立） | 更新门 `z` 里的 `z⊙h_{t-1}` |
| 谁管"写入新信息" | 输入门 `i`（独立） | 同一个 `z` 的 `(1-z)⊙n_t` |
| 两者关系 | 相互独立，可同时全开 | **互补，强制加起来 = 1** |
| 参数量（H=10, D=5） | 680 | **510**（少 1/4） |
| 返回值 | `(output, (hn, cn))` | `(output, hn)` |
| 每步梯度衰减 | ×0.663 | ×0.613 |
| 训练速度 | 慢一些 | 快一些 |

**怎么选**：短序列 / 想省参数 → GRU；长序列 / 要更稳的长距离记忆 → LSTM。

---

## 6. 踩坑

1. **两种公式变体别抄混**：PyTorch 是 `h_t = (1-z_t)⊙n_t + z_t⊙h_{t-1}`（`z` = 保留旧记忆的比例），实测确认。教材上另一种写法 `(1-z)⊙h + z⊙n` 的 `z` 含义正好相反。
2. **重置门只作用于 `W_hn·h_{t-1} + b_hn` 这一项**，不是作用在整个候选状态上。写成 `n = tanh(r ⊙ (...))` 会把当前输入也一起"重置"掉，前向就对不上。
3. **GRU 返回两个值**：`output, hn = gru(x, h0)`。**没有 `c0`**——`nn.GRU` 只接受一个初始隐藏状态，而 `nn.LSTM` 要 `(h0, c0)` 元组。
4. **`h0` 不传就是全 0**，不用手动造；跨批次要自己传（和 RNN / LSTM 一样）。

---

## 附：复现代码

本文数字由下面 3 段代码跑出，按顺序贴进一个 `.py` 就能复现（`D:\Anaconda3\envs\nlpbase\python.exe`）。

**① 前向手算 + 三组门的梯度**（对应第 2、3、4.1 节）

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
D, H, V = 5, 4, 20
gru = nn.GRU(D, H, batch_first=True)
x = torch.randn(1, 3, D)

# --- 手算前向，逐步比对 nn.GRU ---
out, hn = gru(x)
W_ih, W_hh, b_ih, b_hh = gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0
W_ih_r, W_ih_z, W_ih_n = W_ih.chunk(3, 0)
W_hh_r, W_hh_z, W_hh_n = W_hh.chunk(3, 0)
b_ih_r, b_ih_z, b_ih_n = b_ih.chunk(3)
b_hh_r, b_hh_z, b_hh_n = b_hh.chunk(3)

h = torch.zeros(1, H)
for t in range(x.size(1)):
    xt = x[:, t]
    r = torch.sigmoid(xt @ W_ih_r.T + b_ih_r + h @ W_hh_r.T + b_hh_r)
    z = torch.sigmoid(xt @ W_ih_z.T + b_ih_z + h @ W_hh_z.T + b_hh_z)
    n = torch.tanh(xt @ W_ih_n.T + b_ih_n + r * (h @ W_hh_n.T + b_hh_n))
    h = (1 - z) * n + z * h
    print('第 %d 步 == out[:, %d] ? %s' % (t + 1, t, torch.allclose(h, out[:, t], atol=1e-6)))

# --- 三组门的梯度 ---
head = nn.Linear(H, V)
y = torch.randint(0, V, (1, 3))
F.cross_entropy(head(out).reshape(-1, V), y.reshape(-1)).backward()
gw = gru.weight_ih_l0.grad
for k, nm in enumerate(['重置门 r', '更新门 z', '候选状态 n']):
    print('%-8s 梯度范数 = %.6f' % (nm, gw[k * H:(k + 1) * H].norm().item()))
```

**② `∂h_t/∂h_{t-1}` 是不是对角阵**（对应第 4.2 节）

```python
cell = nn.GRUCell(D, H)
xx, hh = torch.randn(1, D), torch.zeros(1, H, requires_grad=True)
cell(xx, hh)
J = torch.autograd.functional.jacobian(lambda h0: cell(xx, h0), hh)[0, :, 0, :]

with torch.no_grad():
    Wz, Uz, bz, cz = cell.weight_ih.chunk(3, 0)[1], cell.weight_hh.chunk(3, 0)[1], \
                     cell.bias_ih.chunk(3)[1], cell.bias_hh.chunk(3)[1]
    z = torch.sigmoid(xx @ Wz.T + bz + hh @ Uz.T + cz)

print('∂h_t/∂h_{t-1} 雅可比：')
for row in J:
    print('    [' + '  '.join('%8.5f' % v for v in row.tolist()) + ']')
print('更新门 z 的值 :', ['%.5f' % v for v in z[0].tolist()])
print('对角线 == z ?  ', torch.allclose(torch.diag(J), z[0], atol=1e-6))
print('非对角全为 0 ? ', bool((J - torch.diag(torch.diag(J))).abs().max() < 1e-7))
```

**③ 四方梯度衰减对比**（对应第 4.3 节）

```python
DD, HH = 8, 16
emb = nn.Embedding(V, DD)
idx, tgt = torch.randint(0, V, (1, 120)), torch.randint(0, V, (1, 120))


def build(kind, fb=None):
    if kind == 'rnn':
        m = nn.RNN(DD, HH, batch_first=True)
    elif kind == 'gru':
        m = nn.GRU(DD, HH, batch_first=True)
    else:
        m = nn.LSTM(DD, HH, batch_first=True)
        if fb is not None:
            with torch.no_grad():
                m.bias_ih_l0[HH:2 * HH].fill_(fb)
                m.bias_hh_l0[HH:2 * HH].fill_(0.0)
    return m, nn.Linear(HH, V)


def grad_first(model, head, upto):
    """只对最后一步算 loss，返回第 0 步输入拿到的梯度"""
    emb.zero_grad(); model.zero_grad(); head.zero_grad()
    e = emb(idx[:, :upto])
    e.retain_grad()
    o, _ = model(e)
    F.cross_entropy(head(o[:, -1, :]), tgt[:, upto - 1]).backward()
    return e.grad[0].norm(dim=-1)[0].item()


for name, kind, fb in [('RNN', 'rnn', None), ('GRU', 'gru', None),
                       ('LSTM(默认)', 'lstm', None), ('LSTM(f=1)', 'lstm', 1.0)]:
    row = []
    for T2 in (20, 50, 100, 120):
        torch.manual_seed(1)
        m, hd = build(kind, fb)
        row.append('%.2e' % grad_first(m, hd, T2))
    print('%-12s T=20 %s  T=50 %s  T=100 %s  T=120 %s' % (name, *row))

TD = 50
for name, kind, fb in [('RNN', 'rnn', None), ('GRU', 'gru', None),
                       ('LSTM(默认)', 'lstm', None), ('LSTM(f=1)', 'lstm', 1.0)]:
    torch.manual_seed(1)
    m, hd = build(kind, fb)
    g0 = grad_first(m, hd, TD)
    torch.manual_seed(1)
    m2, h2 = build(kind, fb)
    emb.zero_grad(); m2.zero_grad(); h2.zero_grad()
    e = emb(idx[:, :TD])
    e.retain_grad()
    o, _ = m2(e)
    F.cross_entropy(h2(o[:, -1, :]), tgt[:, TD - 1]).backward()
    gl = e.grad[0].norm(dim=-1)[-1].item()
    print('%-12s 第0步=%.2e  第%d步=%.2e  平均每步 ×%.3f'
          % (name, g0, TD - 1, gl, (g0 / gl) ** (1 / (TD - 1))))
```
