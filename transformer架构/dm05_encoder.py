from dm04_encoder_layer import *

class Encoder(nn.Module):
    """
    编码器
    由多个编码器层堆叠而成
    """
    def __init__(self, layer, N):
        """
        :param layer: 编码器层实例
        :param N: 编码器层的数量
        """
        super().__init__()
        self.layers = clones(layer, N)  # 创建N个编码器层
        self.norm = LayerNorm(layer.d_model)  # 层归一化

    def forward(self, x, mask=None):
        """
        前向传播
        :param x: 输入张量 (B, L, d_model)
        :param mask: 掩码 (B, L, L)
        :return: 输出张量 (B, L, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)  # 依次通过每个编码器层
        return self.norm(x)  # 最后进行层归一化

def use_encoder():
    """
    使用编码器
    """
    position_x = use_position()
    encoder_layer = EncoderLayer(d_model=512, heads=8, d_ff=2048, dropout=0.1)
    encoder = Encoder(encoder_layer, N=6)  # 创建一个包含6个编码器层的编码器
    print('编码器:', encoder)
    print('位置编码输出:', position_x.shape)  # (2, 10, 512)
    mask = torch.ones(position_x.size(0), position_x.size(1), position_x.size(1))  # 假设没有掩码(全部遮挡)
    print('掩码:', mask)
    print('掩码形状:', mask.shape)  # (2, 10, 10)
    result = encoder(position_x, mask)  # 使用编码器
    print('编码器输出形状:', result.shape)  # (2, 10, 512)
    print('编码器输出:', result)  # 输出张量
    return result


if __name__ == "__main__":
    use_encoder()