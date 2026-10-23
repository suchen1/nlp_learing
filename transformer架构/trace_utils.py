# -*- coding: utf-8 -*-
"""模型调试工具：看清"每一层在干什么"。

用途
----
1. trace(model, x)      —— 跑一次 forward，打印每层的输入/输出形状 + 参数量
2. param_summary(model) —— 只看参数分布（谁最"重"）
3. Wrap(module)         —— 把"需要多个参数"的模块包成单输入，方便 trace

典型用法
--------
    from trace_utils import trace, Wrap

    model.eval()                       # ★ 一定要先 eval，否则 dropout 会随机
    trace(model, x)                    # x 是模型 forward 的第一个参数

    # 多参数模块（如 Multi_Head_Attention(q,k,v)）包一层
    trace(Wrap(Multi_Head_Attention(512, 8)), x)

    # 要传 mask
    trace(Wrap(Multi_Head_Attention(512, 8)), x, mask=None)
"""
import torch
import torch.nn as nn

__all__ = ['trace', 'param_summary', 'Wrap']


def _shape(t):
    """把任意东西描述成形状字符串（Tensor / tuple / PackedSequence 都能处理）"""
    if isinstance(t, torch.Tensor):
        return str(tuple(t.shape))
    if isinstance(t, (tuple, list)):
        return '(' + ', '.join(_shape(i) for i in t) + ')'
    if hasattr(t, 'data') and hasattr(t, 'batch_sizes'):      # PackedSequence
        return 'Packed(%s)' % (tuple(t.data.shape),)
    return type(t).__name__


def trace(model, *args, show_params=True, collapse=True, max_depth=None, **kwargs):
    """跑一次 forward，用【树形】打印每个模块的 输入/输出形状 + 参数量。

    相比平铺列表，树形输出能一眼看出层级关系；同类型的重复层会自动折叠。

    :param model:  任意 nn.Module
    :param args:   传给 model.forward 的位置参数（通常就是输入张量）
    :param show_params: 是否显示参数量
    :param collapse: 同类型重复的兄弟模块是否折叠成 ×N（只展开第一个）
    :param max_depth: 最多展开几层（None = 全部）
    :param kwargs: 传给 model.forward 的关键字参数（如 mask=...）
    :return: {模块全名: (输入形状, 输出形状)}, 以及总参数量

    ★ 内部会临时 model.eval()，结束后恢复原状态，所以 dropout 不会干扰。

    读法::

        model : Transformer                        ← 根节点
        ├─ encoder : Encoder          (2,10,512)→(2,10,512)
        │  ├─ layers : ModuleList
        │  │  └─ [0] : EncoderLayer   ...   ×6（只展开第1个）
        │  └─ norm : LayerNorm
        └─ output_layer : Output

    注意：SublayerConnection 是"回调式"设计，它不持有子层，
    而是在 forward(x, sublayer) 里接收函数 —— 所以对象树上
    sublayers 和 self_attn 是【兄弟】，包裹关系不在树里。
    """
    # ── 1. 跑一次 forward，用 hook 收集每个模块的形状 ──
    shapes = {}

    def make_hook(name):
        def hook(mod, inp, out):
            shapes[name] = (_shape(inp[0]) if inp else '-', _shape(out))
        return hook

    was_training = model.training
    model.eval()
    handles = [m.register_forward_hook(make_hook(n))
               for n, m in model.named_modules() if n != '']
    try:
        with torch.no_grad():
            model(*args, **kwargs)
    finally:
        for h in handles:
            h.remove()                       # ★ 必须移除，否则记录会累积
        model.train(was_training)            # 恢复原来的训练/推理状态

    total = sum(p.numel() for p in model.parameters())
    print('总参数 %s 个（%.2f MB float32）'
          % (format(total, ','), total * 4 / 1024 / 1024))
    print()

    # ── 2. 按模块树递归打印 ──
    def rec(mod, full_name, indent, is_last, label, note=''):
        conn = '└─ ' if is_last else '├─ '
        n_params = sum(p.numel() for p in mod.parameters())
        own = sum(p.numel() for p in mod.parameters(recurse=False))

        line = '%s%s%s : %s' % (indent, conn, label, type(mod).__name__)
        shp = shapes.get(full_name)
        if shp:
            line += '  %s → %s' % shp
        if show_params and n_params:
            # 星号标出"参数就在这一层"（叶子），没星号说明在子模块里
            line += '  %s %s' % ('★' if own == n_params else ' ', format(n_params, ','))
        if note:
            line += '   ' + note
        print(line)

        kids = list(mod.named_children())
        if not kids:
            return
        depth = indent.count('│') + indent.count('   ')
        if max_depth is not None and depth >= max_depth:
            print('%s   …（还有 %d 个子模块，已省略）'
                  % (indent + ('   ' if is_last else '│  '), len(kids)))
            return

        sub = indent + ('   ' if is_last else '│  ')

        # 同类型重复 → 折叠成 ×N，只展开第一个
        if collapse and len(kids) > 1 and len({type(c) for _, c in kids}) == 1:
            first_name, first_mod = kids[0]
            rec(first_mod,
                full_name + '.' + first_name if full_name else first_name,
                sub, True, first_name or '[0]',
                '×%d 个（只展开第 1 个，其余完全相同）' % len(kids))
            return

        for i, (name, child) in enumerate(kids):
            child_full = (full_name + '.' + name) if full_name else name
            rec(child, child_full, sub, i == len(kids) - 1, name or '[%d]' % i)

    rec(model, '', '', True, 'model')

    print()
    print('★ 表示参数就在这一层（叶子模块）；没有 ★ 的是容器，参数在子模块里')
    return shapes


def param_summary(model):
    """只列有直接参数的模块，按参数量从大到小排（看谁最"重"）。"""
    rows = []
    for name, mod in model.named_modules():
        if name == '':
            continue
        own = sum(p.numel() for p in mod.parameters(recurse=False))
        if own:
            rows.append((name, type(mod).__name__, own))
    rows.sort(key=lambda r: -r[2])

    total = sum(p.numel() for p in model.parameters())
    print('  %-36s %-22s %-12s %s' % ('模块路径', '类型', '参数量', '占比'))
    print('  ' + '-' * 88)
    for name, typ, n in rows:
        print('  %-36s %-22s %-12s %.1f%%'
              % (name, typ, format(n, ','), n / total * 100 if total else 0))
    print('  ' + '-' * 88)
    print('  总计 %s 个参数（%.2f MB）' % (format(total, ','), total * 4 / 1024 / 1024))


class Wrap(nn.Module):
    """把"需要多个参数"的模块包成单输入，方便 trace。

    Wrap(m)                -> forward(x) = m(x, x, x)       自注意力 / 交叉注意力同源
    Wrap(m, mode='one')    -> forward(x) = m(x)             普通单输入模块
    Wrap(m, mode='mask')   -> forward(x, mask=None) = m(x, x, x, mask)

    ★ 注意：被包的模块 m 是【注册为子模块】的，所以参数会正常出现在
      parameters() 里 —— 和 lambda 里现做完全不同。
    """
    def __init__(self, m, mode='self'):
        super().__init__()
        assert mode in ('self', 'one', 'mask'), "mode 只能是 'self' / 'one' / 'mask'"
        self.m = m
        self.mode = mode

    def forward(self, x, mask=None):
        if self.mode == 'one':
            return self.m(x)
        if self.mode == 'mask':
            return self.m(x, x, x, mask)
        return self.m(x, x, x)


if __name__ == '__main__':
    # 自测：造一个两层的小模型验证工具本身没问题
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.norm = nn.LayerNorm(16)
            self.fc2 = nn.Linear(16, 8)

        def forward(self, x):
            return self.fc2(torch.relu(self.norm(self.fc1(x))))

    m = Tiny()
    print('=== trace ===')
    trace(m, torch.randn(2, 5, 8))
    print()
    print('=== param_summary ===')
    param_summary(m)
    print()
    print('=== Wrap（单输入模块用 mode="one"）===')
    trace(Wrap(nn.Linear(8, 8), mode='one'), torch.randn(2, 5, 8))
    print()
    print('=== Wrap（自注意力模块用默认 self）===')

    class TinyAttn(nn.Module):
        """假装是多头注意力：接受 (q, k, v)"""
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, q, k, v):
            return self.lin(q + k + v)

    trace(Wrap(TinyAttn()), torch.randn(2, 5, 8))
