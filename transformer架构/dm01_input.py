import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    """嵌入层
    :param vocab_size: 词汇表大小
    :param d_model: 嵌入维度
    """
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model  

    def forward(self, x):
        # 对输入的词索引进行嵌入，并乘以嵌入维度的平方根（用于缩放）
        return self.embed(x) * math.sqrt(self.d_model)