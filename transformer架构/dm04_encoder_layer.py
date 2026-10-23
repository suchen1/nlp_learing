from dm03_encoder_sublayer import *

class EncoderLayer(nn.Module):
    """
    编码器层
    包含多头注意力机制和前馈神经网络
    """
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        """编码器层,包含多头注意力机制和前馈神经网络
        
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
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask=None):
        """
        前向传播
        :param x: 输入张量 (B, L, d_model)
        :param mask: 掩码 (B, L, L)
        :return: 输出张量 (B, L, d_model)
        """
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))  # 多头注意力子层
        x = self.sublayers[1](x, self.feed_forward)  # 前馈神经网络子层
        return x

def use_encoder_layer():
    position_x = use_position()
    encoder_layer = EncoderLayer(d_model=512, heads=8, d_ff=2048, dropout=0.1)
    print('编码器层:', encoder_layer)
    print('位置编码输出:', position_x.shape)  # (2, 10, 512)
    mask = torch.zeros(position_x.size(0), position_x.size(1), position_x.size(1))  # 假设有掩码(全部遮挡)
    print('掩码:', mask)
    print('掩码形状:', mask.shape)  # (2, 10, 10)
    result = encoder_layer(position_x, mask)  # 使用编码器层
    print('编码器层输出形状:', result.shape)  # (2, 10, 512)
    print('编码器层输出:', result)  # 输出张量
    return result


if __name__ == "__main__":
    use_encoder_layer()