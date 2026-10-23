from dm06_decoder_layer import *

class Decoder(nn.Module):
    def __init__(self, layer, N):
        """解码器 由多个解码器层堆叠而成
        
        :param layer: 解码器层实例
        :param N: 解码器层的数量
        """
        super().__init__()
        self.layers = clones(layer, N)  # 创建N个解码器层
        self.norm = LayerNorm(layer.d_model)  # 层归一化

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        前向传播
        :param x: 输入张量 (B, L_tgt, d_model)
        :param memory: 编码器输出 (B, L_src, d_model)
        :param src_mask: 源序列掩码(编码器-解码器) (B, L_src, L_src)
        :param tgt_mask: 目标序列掩码 (B, L_tgt, L_tgt)
        :return: 输出张量 (B, L_tgt, d_model)
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)  # 依次通过每个解码器层
        return self.norm(x)  # 最后进行层归一化