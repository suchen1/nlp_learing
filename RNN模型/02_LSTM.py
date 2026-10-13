import torch
import torch.nn as nn

def dm01_LSTM():
    # 定义LSTM模型 参数一：输入特征的维度 参数二：隐藏状态的维度 参数三：LSTM层数 参数四：是否将batch_size放在第一维 参数五：是否使用双向LSTM
    lstm = nn.LSTM(input_size=5, hidden_size=10, num_layers=1, batch_first=True,bidirectional=False)

    input = torch.randn(1, 4, 5)  # batch_size=1, seq_len=4, input_size=5
    print(f'Input shape:{input.shape},Input:\n{input}')
    h0 = torch.randn(1, 1, 10)  # num_layers=1, batch_size=1, hidden_size=10
    c0 = torch.randn(1, 1, 10)  # num_layers=1, batch_size=1, hidden_size=10
    print(f'H0 shape:{h0.shape},H0:\n{h0}')
    print(f'C0 shape:{c0.shape},C0:\n{c0}')
    output, (h1, c1) = lstm(input, (h0, c0))
    print(f"Output shape:{output.shape},Output:\n{output}")
    print(f"H1 shape:{h1.shape},H1:\n{h1}")
    print(f"C1 shape:{c1.shape},C1:\n{c1}")

if __name__ == "__main__":
    dm01_LSTM()