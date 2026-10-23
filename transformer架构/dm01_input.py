import os
os.system('cls' if os.name == 'nt' else 'clear')     # 先清屏再跑
import torch
import torch.nn as nn
import math
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        """嵌入层
        :param vocab_size: 词汇表大小
        :param d_model: 嵌入维度
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model  

    def forward(self, x):
        # 对输入的词索引进行嵌入，并乘以嵌入维度的平方根（用于缩放）
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100,dropout=0.1):
        """位置编码层
        :param d_model: 嵌入维度
        :param max_len: 最大序列长度
        :param dropout: dropout率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建一个位置编码矩阵，形状为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 位置索引 [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 计算分母项
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦函数
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦函数
        pe = pe.unsqueeze(0)  # 形状为 (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 将位置编码注册为缓冲区

    def forward(self, x): # x: [batch_size, seq_len, d_model]
        return self.dropout(x + self.pe[:, :x.size(1)])


def use_position():
    """
    使用位置编码
    """
    vocab_size = 1000
    d_model = 512
    max_len = 1000
    embedding_layer = Embedding(vocab_size, d_model)
    positional_encoding_layer = PositionalEncoding(d_model, max_len)
    x = torch.randint(0, vocab_size, (2, 10))  # 假设输入是一个形状为 (batch_size=2, seq_len=10) 的词索引张量
    print('输入:', x.shape)  # (2, 10)
    embedded_x = embedding_layer(x)  # 嵌入层输出
    output = positional_encoding_layer(embedded_x)  # 添加位置编码
    return output  # 输出形状为 (batch_size=2, seq_len=10, d_model=512)


if __name__ == "__main__":
    use_position()