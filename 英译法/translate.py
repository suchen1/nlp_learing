import re
import os
os.system('cls' if os.name == 'nt' else 'clear')     # 先清屏再跑
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SOS_TOKEN = 0 # 起始符号
EOS_TOKEN = 1 # 结束符号
MAX_LENGTH = 10  # 最大句子长度
data_path = "./data/eng-fra-v2.txt"  # 数据文件路径

def normalize_string(s):
    """将字符串转换为小写并去除非字母字符"""
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", " ", s)
    return s

def read_data(file_path):
    """读取数据文件，构建词汇表和句子对"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        my_pairs = [ [normalize_string(s) for s in line.split('\t')] for line in lines]
        print(my_pairs[4][0])  # 打印第5行的英文句子
        english_words2index = {'SOS': SOS_TOKEN, 'EOS': EOS_TOKEN}  # 英文词汇表
        english_words_count = 2

        french_words2index = {'SOS': SOS_TOKEN, 'EOS': EOS_TOKEN}  # 法语词汇表
        french_words_count = 2

        for pair in my_pairs:
            # 统计英文词汇表
            for word in pair[0].split(' '):
                if word not in english_words2index:
                    english_words2index[word] = english_words_count
                    english_words_count += 1
            # 统计法语词汇表
            for word in pair[1].split(' '):
                if word not in french_words2index:
                    french_words2index[word] = french_words_count
                    french_words_count += 1

        # 构建反向映射表
        english_index2words = {index: word for word, index in english_words2index.items()}
        french_index2words = {index: word for word, index in french_words2index.items()}
        return my_pairs, english_words2index, english_index2words, english_words_count, french_words2index, french_index2words, french_words_count

my_pairs, english_words2index, english_index2words, english_words_count, french_words2index, french_index2words, french_words_count = read_data(data_path)


class My_Dataset(Dataset):
    """自定义数据集类"""
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        """返回数据集的大小"""
        return len(self.pairs)

    def __getitem__(self, idx):
        english_sentence = self.pairs[idx][0]
        french_sentence = self.pairs[idx][1]
        english_indices = [english_words2index[word] for word in english_sentence.split(' ')] + [EOS_TOKEN]
        french_indices = [french_words2index[word] for word in french_sentence.split(' ')] + [EOS_TOKEN]
        return torch.tensor(english_indices, dtype=torch.long).to(device), torch.tensor(french_indices, dtype=torch.long).to(device)

def my_dataloader(batch_size=1, shuffle=True):
    """创建数据加载器"""
    dataset = My_Dataset(my_pairs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class EncoderRNN(nn.Module):
    """编码器RNN"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size # 隐藏层维度
        self.input_size = input_size   # 输入词汇表大小 
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,batch_first=True) # 词嵌入层， 隐藏层维度，仅仅数值一样

    def forward(self, input, hidden):
        """前向传播"""
        embedded = self.embedding(input).unsqueeze(0)  # [seq_len] -> [1, seq_len, hidden_size]
        output, hidden = self.gru(embedded, hidden)    # output: [1, seq_len, hidden_size], hidden: [1, 1, hidden_size]
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)

def test_encoder():
    """测试编码器"""
    input_size = english_words_count
    hidden_size = 256
    encoder = EncoderRNN(input_size, hidden_size).to(device)
    input_tensor = torch.tensor([english_words2index[word] for word in "i am a student".split(' ')] + [EOS_TOKEN], dtype=torch.long).to(device)
    hidden = encoder.init_hidden()
    output, hidden = encoder(input_tensor, hidden)
    print(f"Encoder output shape: {output.shape}, output:\n {output}")
    print(f"Encoder hidden shape: {hidden.shape}, hidden:\n {hidden}")

class DecoderRNN_No_Attention(nn.Module):
    """解码器RNN（无注意力机制）"""
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(0.1)
    def forward(self, input, hidden):
        """前向传播"""
        embedded = self.embedding(input) # [1, seq_len] -> [1, seq_len, hidden_size]
        embedded = self.drop(embedded)  # dropout随即失活10%
        output, hidden = self.gru(embedded, hidden)    # output: [1, seq_len, hidden_size], hidden: [1, 1, hidden_size]
        return output, hidden
if __name__ == "__main__":
    test_encoder()