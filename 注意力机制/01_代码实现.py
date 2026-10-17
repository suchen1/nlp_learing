import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    案例：
    已知QKV:v是内容比如32个单词,每个单词64个特征,k是32个单词的索引,q是查询张量
    我们的任务:输入查询张量q,通过注意力机制来计算如下信息:
        1、查询张量q的注意力权重分布:查询张量q和32个单词相关性(相识度)
        2、查询张量q的结果表示:由一个普通的q升级成一个更强大q
        3、注意:查询张量q查询的目标是谁，就是谁的查询张量。
    例如:查询张量q是来查询单词"我",则q就是"我"的查询张量
"""

class My_Attention(nn.Module):
    def __init__(self, query_dim, key_dim, word_count, value_dim, output_size):
        super(My_Attention, self).__init__()
        self.query_dim = query_dim  # 查询张量q的维度[1, 64]
        self.key_dim = key_dim      # 键张量k的维度[1, 64]
        self.value_dim = value_dim  # 值张量v的维度[32, 64]
        self.output_size = output_size # 输出张量的维度[1, 32]
        self.word_count = word_count # 单词数量 32

        # 注意力权重计算层:将Q和K拼接后，映射到:值序列长度维度.
        self.atten = nn.Linear(self.query_dim + self.key_dim, self.word_count)
        # 注意力输出层:将注意力输出和V拼接后，映射到输出维度.
        self.atten_out = nn.Linear(self.query_dim + self.value_dim, self.output_size)

    def forward(self, q, k, v):
        """
        :param q: 查询张量q,形状[1, 1, 64]
        :param k: 键张量k,形状[1, 1, 64]
        :param v: 值张量v,形状[1, 32, 64]
        :return: 输出张量,形状[1, 32]
        """
        # 1、计算注意力权重分布
        # 将查询张量q和键张量k拼接
        q_k_concat = torch.cat((q[0],k[0]), dim=-1)  # [1, 64] + [1, 64] = [1, 128]
        print(f'q_k_concat shape: {q_k_concat.shape}')  # [1, 128]

        # 通过线性层计算注意力权重
        attention_weights = self.atten(q_k_concat)  # [1, 128] -> [1, 32]
        attention_weights = F.softmax(attention_weights, dim=-1)  # [1, 32]
        print(f'attention_weights shape: {attention_weights.shape}')  # [1, 32]

        # 2、计算注意力输出
        # 将注意力权重与值张量v相乘
        attention_weights = attention_weights.unsqueeze(0)  # [1, 1, 32]
        attention_output = torch.bmm(attention_weights, v)  # [1, 1, 32] * [1, 32, 64] = [1, 1, 64]
        print(f'attention_output shape: {attention_output.shape}')  # [1, 1, 64]
        output = self.atten_out(torch.cat((attention_output[0], q[0]), dim=-1))  # [1, 64] + [1, 64] = [1, 128] -> [1, 32]
        return output


# ══════════════════════════════════════════════════════════════════════════
# 附：真正的「内容比对」版 —— K 是【每个词各一个】，形状 [1, 32, 64]
#     和上面的 My_Attention 放一起对比：同一个任务、两种实现、权重行为完全不同
# ══════════════════════════════════════════════════════════════════════════
class My_Attention_Content(nn.Module):
    """内容比对版：分数由【每个词自己的 key】算出来（点积）。

    和 My_Attention 的根本区别：
        My_Attention          分数 = Linear([q ; k])    ← k 是【一个】共享向量
        My_Attention_Content  分数 = q · k_j / √d       ← 每个词【各有】一个 k_j

    输入：
        q [1, 1, 64]      k [1, 32, 64]      v [1, 32, 64]
    输出：
        [1, output_size]
    """

    def __init__(self, query_dim, value_dim, output_size):
        super().__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.output_size = output_size
        # 评分部分不需要参数（点积本身就是相似度），只保留融合层
        self.atten_out = nn.Linear(self.query_dim + self.value_dim, self.output_size)

    def forward(self, q, k, v):
        # 1、算分数：q 和【每一个】k 做点积  →  [1, 1, 32]
        scores = torch.bmm(q, k.transpose(-2, -1)) / (self.query_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)              # [1, 1, 32]

        # 2、加权求和 V
        attention_output = torch.bmm(attention_weights, v)         # [1, 1, 64]

        # 3、融合 q 和 attention_output
        output = self.atten_out(torch.cat((q[0], attention_output[0]), dim=-1))  # [1,128] -> [1,32]
        return output


if __name__ == '__main__':
    torch.manual_seed(0)
    q = torch.randn(1, 1, 64)
    v = torch.randn(1, 32, 64)

    print('=' * 64)
    print('两种实现的对比')
    print('=' * 64)

    # ── ① 内容比对版：k 是 32 个词各一个 ──
    m2 = My_Attention_Content(64, 64, 32)
    k32 = torch.randn(1, 32, 64)
    out2 = m2(q, k32, v)
    print(f'\n[内容比对版] 输出 {tuple(out2.shape)}')

    k32_b = k32.clone()
    k32_b[0, 5] += 10.0                                    # 只改第 5 个词的 key
    s_a = torch.bmm(q, k32.transpose(-2, -1)).squeeze()
    s_b = torch.bmm(q, k32_b.transpose(-2, -1)).squeeze()
    changed = torch.nonzero((s_a - s_b).abs() > 1e-6).flatten().tolist()
    print(f'  只改第 5 个词的 k  →  变化的分数下标: {changed}   ← 只有它自己')

    # ── ② 位置打分版：k 是一个共享向量 ──
    print()
    m1 = My_Attention(64, 64, 32, 64, 32)
    k1 = torch.randn(1, 1, 64)
    out1 = m1(q, k1, v)
    print(f'\n[位置打分版] 输出 {tuple(out1.shape)}')

    w_a = F.softmax(m1.atten(torch.cat((q[0], k1[0]), dim=-1)), dim=-1)
    w_b = F.softmax(m1.atten(torch.cat((q[0], (k1 + 10.0)[0]), dim=-1)), dim=-1)
    n_changed = int(((w_a - w_b).abs().squeeze() > 1e-6).sum())
    print(f'  改那个唯一的 k  →  变化的权重个数: {n_changed}/32   ← 全都变了')
    print('  （因为这个 k 参与了每一行的计算，压根没有"某一个词"的概念）')

    print()
    print('=' * 64)
    print('结论：')
    print('  位置打分版：分数输入是 [q ; 一个共享的 k]  → 改 k 影响全部位置')
    print('  内容比对版：分数输入是 [q ; k_j]            → 改哪个词的 k 只影响哪个分数')
    print('  Transformer 用的是后者（变长句子才可能成立）')
