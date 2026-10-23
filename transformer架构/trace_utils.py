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


def trace(model, *args, show_params=True, **kwargs):
    """跑一次 forward，打印每一层的输入/输出形状（以及自身参数量）。

    :param model: 任意 nn.Module
    :param args:  传给 model.forward 的位置参数（通常就是输入张量）
    :param show_params: 是否显示参数量列
    :param kwargs: 传给 model.forward 的关键字参数（如 mask=...）
    :return: [(层名, 类型, 输入形状, 输出形状, 自身参数量), ...]

    ★ 内部会 model.eval() 再恢复原状态，所以 dropout 不会干扰。
    """
    rec = []
    was_training = model.training
    model.eval()

    def make_hook(name):
        def hook(mod, inp, out):
            rec.append((
                name,
                type(mod).__name__,
                _shape(inp[0]) if inp else '(无输入)',
                _shape(out),
                sum(p.numel() for p in mod.parameters(recurse=False)),
            ))
        return hook

    handles = [m.register_forward_hook(make_hook(n))
               for n, m in model.named_modules() if n != '']
    try:
        with torch.no_grad():
            model(*args, **kwargs)
    finally:
        for h in handles:
            h.remove()                       # ★ 必须移除，否则记录会累积
        model.train(was_training)            # 恢复原来的训练/推理状态

    if show_params:
        print('  %-34s %-22s %-16s %-16s %s'
              % ('模块', '类型', '输入', '输出', '自身参数'))
        print('  ' + '-' * 104)
    else:
        print('  %-34s %-22s %-16s %s' % ('模块', '类型', '输入', '输出'))
        print('  ' + '-' * 92)

    for name, typ, i_s, o_s, n in rec:
        depth = name.count('.')
        label = '│ ' * depth + name.split('.')[-1]
        if show_params:
            print('  %-34s %-22s %-16s %-16s %s'
                  % (label, typ, i_s, o_s, format(n, ',') if n else ''))
        else:
            print('  %-34s %-22s %-16s %s' % (label, typ, i_s, o_s))

    if show_params:
        print('  ' + '-' * 104)
    else:
        print('  ' + '-' * 92)

    total = sum(p.numel() for p in model.parameters())
    print('  总计：%d 层模块，%s 个参数（%.2f MB float32）'
          % (len(rec), format(total, ','), total * 4 / 1024 / 1024))
    return rec


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
