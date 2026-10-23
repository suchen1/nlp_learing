import os
os.system('cls' if os.name == 'nt' else 'clear')     # 先清屏再跑
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from dm01_input import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dm01_test_triu():
    batch_size, seq_len, d_model = 2, 5, 4
    d = d_model

    # 假装这是注意力：q/k/v 都从 x 变出来
    x = torch.randn(batch_size, seq_len, d_model)        # (2, 5, 4)
    q = k = v = x

    # ① 分数矩阵 (B, L, L)  ← 长度为 5 的那两个维度
    scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(d)
    print('scores:', scores.shape)                        # (2, 5, 5)

    # ② 两种视角的 mask（都行，选一个）
    blocked = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    allowed = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
    print('blocked（禁止=上三角）:\n', blocked.int())
    print('allowed（允许=下三角）:\n', allowed.int())

    # ③ 作用在分数上
    masked = scores.masked_fill(blocked, float('-inf'))           # 视角 1
    # masked = scores.masked_fill(~allowed, float('-inf'))        # 视角 2（等价）

    # ④ 归一化
    attn = F.softmax(masked, dim=-1) # [B, L, L]
    print('attention（上三角必须全 0，每行和为 1）:\n', attn)
    print('每行和:', attn[0].sum(-1))
    print('输出:', torch.bmm(attn, v).shape)              # (2, 5, 4)_masked)


def clones(model,N):
    """
    克隆N个模块
    :param model: 模块
    :param N: 克隆数量
    :return: nn.ModuleList模块列表
    """
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])

def attention(q, k, v, mask=None, dropout=None):
    """
    计算注意力
    :param q: 查询向量 (B, H, L, d_k)
    :param k: 键向量 (B, H, L, d_k)
    :param v: 值向量 (B, H, L, d_v)
    :param mask: 掩码
    :param dropout: dropout层
    :return: 注意力输出和注意力权重
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, L_q, d_k) * (B, H, d_k, L_k) -> (B, H, L_q, L_k)
    if mask is not None:
        # scores:[B, H, L_q, L_k],mask:[B, 1, L_q, L_k]，mask为0的位置表示不允许关注
        scores = scores.masked_fill(mask == 0, float('-inf')) 
    p_attn = F.softmax(scores, dim=-1)  # (B, H, L_q, L_k)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, v), p_attn  # (B, H, L_q, L_k) * (B, H, L_k, d_v) -> (B, H, L_q, d_v) , p_attn

class Multi_Head_Attention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, embd_dim, heads,droupout=0.1):
        """"
        :param embd_dim: 输入输出的维度
        :param heads: 注意力头数
        :param dropout: dropout率
        """
        super().__init__()
        self.d_model = embd_dim
        self.h = heads
        assert embd_dim % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.d_k = embd_dim // heads

        # 线性层
        self.linear_q = nn.Linear(self.d_model, self.d_model)
        self.linear_k = nn.Linear(self.d_model, self.d_model)
        self.linear_v = nn.Linear(self.d_model, self.d_model)
        self.linear_out = nn.Linear(self.d_model, self.d_model)

    def forward(self, q, k, v,mask=None):
        """
        :param q: 查询向量 (B, L_q, d_model)
        :param k: 键向量 (B, L_k, d_model)
        :param v: 值向量 (B, L_v, d_model)
        :param mask: 掩码 (B, L_q, L_k)
        :return: 注意力输出 (B, L_q, d_model)
        """
        batch_size = q.size(0)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, L, L)
        # 线性变换并分头
        q = self.linear_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (B, L, H, d_k) -> (B, H, L, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (B, L, H, d_k) -> (B, H, L, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  # (B, L, H, d_k) -> (B, H, L, d_k)

        # 注意力计算
        output,p_attn = attention(q, k, v, mask=mask)  # (B, H, L_q, d_k), (B, H, L_q, L_k)

        # 输出 
        # [B, H, L_q, d_k] -> [B, L_q, H, d_k] -> [B, L_q, H*d_k] -> [B, L_q, d_model]
        output = output.permute(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        output = self.linear_out(output)  # (B, L_q, d_model)

        return output

class PositionwiseFeedForward(nn.Module):
    """前馈神经网络层
    :param d_model: 输入输出的维度
    :param d_ff: 前馈网络中间层的维度
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: 输入输出的维度
        :param d_ff: 前馈网络中间层的维度
        :param dropout: dropout率
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 先通过第一个线性层，然后经过ReLU激活，再通过第二个线性层，最后应用dropout
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class LayerNorm(nn.Module):
    """层归一化
    :param features: 输入特征的维度
    :param eps: 防止除零的小数
    """
    def __init__(self, features, eps=1e-9):
        """
        :param features: 输入特征的维度
        :param eps: 防止除零的小数
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # 缩放参数
        self.b_2 = nn.Parameter(torch.zeros(features))  # 偏移参数
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # 计算均值
        std = x.std(-1, keepdim=True)    # 计算标准差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2  # 应用归一化公式
    
def use_multihead_attention():
    my_multi_head_attention = Multi_Head_Attention(512,8)
    position_x = use_position()
    print('位置编码输出:', position_x.shape)  # (2, 10, 512)
    print('多头注意力层:', my_multi_head_attention)
    q = k = v = position_x
    result = my_multi_head_attention(q, k, v)
    print('多头注意力输出:', result.shape)  # (2, 10, 512)
    return result

def use_FFN():
    my_ffn = PositionwiseFeedForward(512, 2048)
    output = use_multihead_attention()
    print('前馈神经网络层:', my_ffn)
    result = my_ffn(output)
    print('前馈神经网络输出:', result.shape)  # (2, 10, 512)
    return result

def use_LayerNorm():
    my_layer_norm = LayerNorm(512)
    output = use_FFN()
    result = my_layer_norm(output)
    print('层归一化输出:', result.shape)  # (2, 10, 512)
    return result

if __name__ == "__main__":
    # use_multihead_attention()
    # use_FFN()
    use_LayerNorm()