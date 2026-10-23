from dm05_encoder import *


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        """
        :param d_model: 输入输出的维度
        :param heads: 注意力头数
        :param d_ff: 前馈网络中间层的维度
        :param dropout: dropout率
        """
        super().__init__()
        self.d_model = d_model
        self.h = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.self_attn = Multi_Head_Attention(d_model, heads, dropout)
        self.src_attn = Multi_Head_Attention(d_model, heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers = clones(SublayerConnection(d_model, dropout), 3)
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        前向传播
        :param x: 输入张量 (B, L_tgt, d_model)
        :param memory: 编码器输出 (B, L_src, d_model)
        :param src_mask: 源序列掩码(编码器-解码器) (B, L_src, L_src)
        :param tgt_mask: 目标序列掩码 (B, L_tgt, L_tgt)
        :return: 输出张量 (B, L_tgt, d_model)
        """
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # 自注意力子层
        x = self.sublayers[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))  # 编码器-解码器注意力子层
        x = self.sublayers[2](x, self.feed_forward)  # 前馈神经网络子层
        return x