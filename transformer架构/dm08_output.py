from dm07_decoder import *

class Output(nn.Module):
    def __init__(self, vocab_size, d_model):
        """输出层，将解码器的输出映射到词汇表大小
        :param vocab_size: 词汇表大小
        :param d_model: 嵌入维度
        """
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)  # 输出层，将解码器输出映射到词汇表大小
    def forward(self, x):
        """
        前向传播
        :param x: 解码器输出 (B, L_tgt, d_model)
        :return: 输出张量 (B, L_tgt, vocab_size)
        """
        return self.linear(x)  # 将解码器输出映射到词汇表大小

def use_output():
    """
    使用输出层
    """
    vocab_size = 10
    d_model = 512
    output_layer = Output(vocab_size, d_model)  # 创建输出层实例
    print('输出层:', output_layer)
    decoder_output = torch.rand(2, 2, d_model)  # 假设解码器输出为随机张量 (B=2, L_tgt=10, d_model=512)
    print('解码器输出形状:', decoder_output.shape)  # (2, 10, 512)
    result = output_layer(decoder_output)  # 使用输出层
    print('输出层输出形状:', result.shape)  # (2, 10, vocab_size)
    print('输出层输出:', result)  # 输出张量
    return result

if __name__ == "__main__":
    use_output()