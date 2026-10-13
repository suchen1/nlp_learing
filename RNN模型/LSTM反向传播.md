# LSTM 的反向传播（Day03 P44–P47）

> **本文回答的问题**：LSTM 正向看懂了（三个门 + 记忆细胞），但**几个门的权重矩阵到底是怎么更新的**？为什么说 `c_t` 是"梯度高速公路"？
>
> 文中每个数字都在本机 PyTorch 上实跑得出（环境 `nlpbase`），**复现代码见文末【附：复现代码】**，复制到任意 `.py` 里就能跑。
>
> 课程 P47 的代码在 [`02_LSTM.py`](02_LSTM.py)（`nn.LSTM` 的 `h0` / `c0` 与输出形状）。

---

## 0. 先纠一个数：不是 3 组权重，是 **4 组**

课件图里讲三个门，但实现上是 **4 组**权重——因为"候选记忆"也要一组自己的权重：

| 门 | 符号 | 作用 | 激活函数 |
| --- | --- | --- | --- |
| 输入门 | `i` | 决定哪些**新**信息写进记忆本 | sigmoid |
| 遗忘门 | `f` | 决定记忆本里哪些**旧**信息划掉 | sigmoid |
| 候选记忆 | `g`（课件里写 `C̃`） | 这一步新算出来的内容 | **tanh** |
| 输出门 | `o` | 决定这一步对外吐出哪些记忆 | sigmoid |

**PyTorch 把这 4 组拼成一个矩阵存**。`nn.LSTM(input_size=5, hidden_size=4)` 实测：

```
weight_ih_l0     (16, 5)     4 组门 × 4 维隐藏 = 16 行
weight_hh_l0     (16, 4)
bias_ih_l0       (16,)
bias_hh_l0       (16,)
```

> ⚠️ **门顺序是 `i, f, g, o`**，不是 `i, f, o, g`。用 `gates.chunk(4, dim=1)` 切出来的就是这个顺序。切错了不会报错，只是数值悄悄错——所以下面用"手算能否对上 `nn.LSTM`"来验证。

---

## 1. 前向公式（和课件图一致）

```
i_t = σ( W_i · [h_{t-1}, x_t] + b_i )          输入门
f_t = σ( W_f · [h_{t-1}, x_t] + b_f )          遗忘门
g_t = tanh( W_g · [h_{t-1}, x_t] + b_g )       候选记忆
o_t = σ( W_o · [h_{t-1}, x_t] + b_o )          输出门

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t                记忆细胞（⊙ 是逐元素相乘）
h_t = o_t ⊙ tanh(c_t)                          对外输出
```

手写实现（对照上面 6 行看）：

```python
import torch, torch.nn as nn

torch.manual_seed(0)
D, H = 5, 4
lstm = nn.LSTM(D, H, num_layers=1, batch_first=True)
x = torch.randn(1, 3, D)
out, (hn, cn) = lstm(x)

W_ih, W_hh = lstm.weight_ih_l0, lstm.weight_hh_l0
b_ih, b_hh = lstm.bias_ih_l0, lstm.bias_hh_l0

h, c = torch.zeros(1, H), torch.zeros(1, H)
for t in range(3):
    gates = x[:, t] @ W_ih.T + b_ih + h @ W_hh.T + b_hh   # (1, 16)
    i, f, g, o = gates.chunk(4, dim=1)                     # ← 顺序 i,f,g,o
    i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
    g = torch.tanh(g)
    c = f * c + i * g
    h = o * torch.tanh(c)
    print(t, torch.allclose(h, out[:, t], atol=1e-6))
```

实测输出：

```
第 1 步 手算 h == out[:, 0] ? True
第 2 步 手算 h == out[:, 1] ? True
第 3 步 手算 h == out[:, 2] ? True
末步 h == hn[0] ? True   c == cn[0] ? True
```

**门顺序和公式都对上了**，可以拿这份代码当"可执行版公式"。

---

## 2. 反向传播走两条路

### 路线 A：沿 `h` 反传 —— 和普通 RNN 一样

```
h_t → h_{t-1} 这条路上要穿过：输出门 σ、tanh、以及 W_hh 矩阵
```

### 路线 B：沿 `c` 反传 —— **这才是 LSTM 的关键**

```
∂c_t / ∂c_{t-1} = diag(f_t)
```

**只乘一个自己选的系数，不乘权重矩阵，也不经过 tanh 的导数。** `nn.LSTMCell` + `torch.autograd.functional.jacobian` 实测的雅可比矩阵：

```
∂c_t/∂c_{t-1} 雅可比:
  [ 0.68539   0.00000   0.00000   0.00000]
  [ 0.00000   0.33751   0.00000   0.00000]
  [ 0.00000   0.00000   0.73105   0.00000]
  [ 0.00000   0.00000   0.00000   0.28154]

遗忘门 f 的值 : ['0.68539', '0.33751', '0.73105', '0.28154']
对角线 == f ?    True
非对角全为 0 ?   True
```

**对照一下就明白"高速公路"是什么意思了**：

| | 每回传一步要乘的东西 | 后果 |
| --- | --- | --- |
| RNN | `W_hhᵀ`（**矩阵**）× tanh 导数 | 反复连乘矩阵 → 很快消失 / 爆炸 |
| LSTM 的 c 路线 | `f_i`（**一个标量**，∈ (0,1)） | 衰减慢得多，且每个记忆单元**独立**控制 |

---

## 3. 四个门的权重怎么更新：一次 backward，梯度分给每一组

只对最后一步算 loss，`backward()` 之后把 `weight_ih_l0.grad` 按门切开：

```
输入门 i    梯度范数 = 0.030090
遗忘门 f    梯度范数 = 0.008038
候选记忆 g   梯度范数 = 0.146673
输出门 o    梯度范数 = 0.015487
weight_hh_l0 整体 = 0.015777    bias 整体 = 0.043945
```

**四组门的梯度全都非零**，说明三件事：

1. **loss 的梯度同时分给了 4 组门**（外加 4 个 bias），不存在"只更新某一个门"。
2. **和 RNN 一样：同一套权重在序列每一步被复用**。T 个时间步各自贡献一份梯度，`backward()` 把它们**累加**到同一个 `.grad` 上。
3. `optimizer.step()` 执行一次，就把 `weight_ih / weight_hh / bias_ih / bias_hh` **一次性全部更新**。

> **一句话**：门多 ≠ 更新次数多。参数变成 RNN 的 4 倍，但更新时机完全一样——**一个 batch 一次 `step()`**。

---

## 4. 实测：梯度到底能回传多远

测法：**只对最后一步算 loss**（这样第 0 步的梯度必须一路走回来），看第 0 步还能拿到多少梯度。序列长度分别取 20 / 50 / 100 / 120：

| 序列长度 | RNN | LSTM（默认初始化） | LSTM（遗忘门偏置 = 1） |
| --- | --- | --- | --- |
| T=20 | 7.71e-07 | 2.10e-05 | 1.88e-03 |
| T=50 | 5.37e-15 | 1.53e-11 | 6.76e-06 |
| T=100 | **0.00e+00**（下溢） | 1.65e-20 | 1.37e-09 |
| T=120 | **0.00e+00**（下溢） | **0.00e+00**（下溢） | 1.58e-11 |

平均**每往前回传一步**，梯度乘多少（T=50 实测）：

| 模型 | 每步衰减系数 |
| --- | --- |
| RNN | × **0.530** |
| LSTM（默认初始化） | × 0.644 |
| LSTM（遗忘门偏置 = 1） | × **0.839** |

**两个能直接用的结论**：

1. **LSTM 强在"衰减慢"**：T=100 时 RNN 的梯度已经下溢成 0，LSTM 还有 1.65e-20；把遗忘门偏置设成 1 之后是 1.37e-09——比 RNN 高约 9 个数量级。
2. **遗忘门偏置初始化为 1（`b_f = 1`）是真实有效的工程技巧**：每步衰减从 0.644 改善到 0.839。含义就是"让遗忘门默认倾向于**记住**"，这是原始 LSTM 论文及后续实践里的标准做法。

```python
# 把遗忘门偏置设成 1（第 2 组 = f，PyTorch 门顺序 i,f,g,o）
H = lstm.hidden_size
with torch.no_grad():
    lstm.bias_ih_l0[H:2 * H].fill_(1.0)
    lstm.bias_hh_l0[H:2 * H].fill_(0.0)
```

---

## 5. 和 RNN 的对照表

| | RNN | LSTM |
| --- | --- | --- |
| 隐藏状态 | 只有 `h_t` | `h_t` **+ `c_t`**（两条线） |
| 权重矩阵个数 | `weight_ih`、`weight_hh` | 还是 2 个，但形状是 **4 倍**（4 组门拼一起） |
| 沿时间回传 | 每步乘 `W_hhᵀ` + tanh 导数 | `h` 路线同 RNN；**`c` 路线只乘 `f`（对角标量）** |
| T=100 时第 0 步梯度 | 下溢为 0 | 1.65e-20（默认）/ 1.37e-09（f 偏置=1） |
| 参数怎么更新 | 一次 backward → 累加 → `step()` 全更新 | **完全一样** |

---

## 6. 踩坑

1. **门顺序是 `i, f, g, o`**，不是 `i, f, o, g`。按错顺序手动实现，前向对不上 `nn.LSTM` 且**不报错**。
2. **`c_t` 不对外输出**。`nn.LSTM` 返回 `(output, (h_n, c_n))`：`output` 是每个时间步的 `h_t`（顶层），`c_n` 才是每层最后一步的 `c_t`。`c` 是内部记忆，只在"接着算下一段"时才用。
3. **别把 `h_t` 和 `c_t` 搞混**：`h_t = o_t ⊙ tanh(c_t)`，是 `c_t` 被输出门过滤后的结果。`c_t` 没有范围限制（可以一路累加），`h_t` 被 tanh 压回 (-1, 1)。
4. **遗忘门偏置默认不是 1**，是随机小值（实测门值 ≈ 0.28~0.73）。长序列任务建议手动设成 1。

---

## 附：复现代码

本文数字由下面 4 段代码跑出，按顺序贴进一个 `.py` 就能复现（`D:\Anaconda3\envs\nlpbase\python.exe`）。

**① 四组门的梯度**（对应第 3 节）

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
D, H, V = 5, 4, 20
lstm = nn.LSTM(D, H, batch_first=True)
x = torch.randn(1, 3, D)
out, (hn, cn) = lstm(x)

head = nn.Linear(H, V)
y = torch.randint(0, V, (1, 3))
F.cross_entropy(head(out).reshape(-1, V), y.reshape(-1)).backward()

gw = lstm.weight_ih_l0.grad
for k, nm in enumerate(['输入门 i', '遗忘门 f', '候选记忆 g', '输出门 o']):
    print('%-8s 梯度范数 = %.6f' % (nm, gw[k * H:(k + 1) * H].norm().item()))
```

**② `∂c_t/∂c_{t-1}` 是不是对角阵 `f`**（对应第 2 节）

```python
cell = nn.LSTMCell(D, H)
xx, hh = torch.randn(1, D), torch.zeros(1, H)
cc = torch.zeros(1, H, requires_grad=True)
cell(xx, (hh, cc))
J = torch.autograd.functional.jacobian(lambda c0: cell(xx, (hh, c0))[1], cc)[0, :, 0, :]

gates = xx @ cell.weight_ih.T + cell.bias_ih + hh @ cell.weight_hh.T + cell.bias_hh
f_val = torch.sigmoid(gates.chunk(4, dim=1)[1])[0]
print('对角线 == f ?  ', torch.allclose(torch.diag(J), f_val, atol=1e-6))
print('非对角全为 0 ? ', bool((J - torch.diag(torch.diag(J))).abs().max() < 1e-7))
```

**③ 梯度能回传多远**（对应第 4 节的表）

```python
DD, HH = 8, 16
emb = nn.Embedding(V, DD)
idx, tgt = torch.randint(0, V, (1, 120)), torch.randint(0, V, (1, 120))


def build(kind, forget_bias=None):
    if kind == 'rnn':
        m = nn.RNN(DD, HH, batch_first=True)
    else:
        m = nn.LSTM(DD, HH, batch_first=True)
        if forget_bias is not None:                       # 遗忘门偏置设成 1
            with torch.no_grad():
                m.bias_ih_l0[HH:2 * HH].fill_(forget_bias)
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


for name, kind, fb in [('RNN', 'rnn', None), ('LSTM(默认)', 'lstm', None), ('LSTM(f偏置=1)', 'lstm', 1.0)]:
    row = []
    for T2 in (20, 50, 100, 120):
        torch.manual_seed(1)
        m, head = build(kind, fb)
        row.append('%.2e' % grad_first(m, head, T2))
    print('%-14s T=20 %s  T=50 %s  T=100 %s  T=120 %s' % (name, *row))
```

**④ 每步衰减系数**（对应第 4 节的表，用 T=50）

```python
T_DECAY = 50
for name, kind, fb in [('RNN', 'rnn', None), ('LSTM(默认)', 'lstm', None), ('LSTM(f偏置=1)', 'lstm', 1.0)]:
    torch.manual_seed(1)
    m, head = build(kind, fb)
    g0 = grad_first(m, head, T_DECAY)                    # 第 0 步

    torch.manual_seed(1)
    m2, h2 = build(kind, fb)
    emb.zero_grad(); m2.zero_grad(); h2.zero_grad()
    e = emb(idx[:, :T_DECAY])
    e.retain_grad()
    o, _ = m2(e)
    F.cross_entropy(h2(o[:, -1, :]), tgt[:, T_DECAY - 1]).backward()
    gl = e.grad[0].norm(dim=-1)[-1].item()               # 最后一步

    per = (g0 / gl) ** (1 / (T_DECAY - 1))
    print('%-14s 第0步=%.2e  第%d步=%.2e  平均每步 ×%.3f' % (name, g0, T_DECAY - 1, gl, per))
```
