from dm01_input import *
from dm02_encoder_element import *

class MY_SublayerConnection(nn.Module):
    """
    子层连接模块
    包含残差连接和层归一化
    """
    def __init__(self, d_model, dropout=0.1):
        """
        :param d_model: 输入输出的维度
        :param dropout: dropout率
        """
        super().__init__()
        self.d_model = d_model
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        前向传播
        :param x: 输入张量 (B, L, d_model)
        :param sublayer: 子层函数
        :return: 输出张量 (B, L, d_model)
        """
        return self.norm(x + self.dropout(sublayer(x))) # 残差连接 + 层归一化

def use_sublayer_connection():
    position_x = use_position()
    my_sublayer_connection1 = MY_SublayerConnection(512)
    print('子层连接模块1:', my_sublayer_connection1)
    result = my_sublayer_connection1(position_x, lambda x: Multi_Head_Attention(512,8)(x, x, x))  # 使用多头注意力作为子层
    print('子层连接1输出:', result.shape)  # (2, 10, 512)
    my_sublayer_connection2 = MY_SublayerConnection(512)
    print('子层连接模块2:', my_sublayer_connection2)
    result2 = my_sublayer_connection2(result, lambda x: PositionwiseFeedForward(512, 2048)(x))  # 使用前馈神经网络作为子层
    print('子层连接2输出:', result2.shape)  # (2, 10, 512)
    return result2




if __name__ == "__main__":
    use_sublayer_connection()