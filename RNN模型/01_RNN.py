import torch.nn as nn
import torch

rnn = nn.RNN(input_size=5, hidden_size=10, num_layers=1, batch_first=True)

input = torch.randn(1, 4, 5)  # batch_size=3, seq_len=4, input_size=5
print(f'Input shape:{input.shape},Input:\n{input}')
h0 = torch.randn(1, 1, 10)  # num_layers=1, batch_size=3, hidden_size=10
print(f'H0 shape:{h0.shape},H0:\n{h0}')
output, h1 = rnn(input, h0)
print(f"Output shape:{output.shape},Output:\n{output}")
print(f"H1 shape:{h1.shape},H1:\n{h1}")
