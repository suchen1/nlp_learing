import math
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
    def __init__(self, input_size, hidden_size, embd_dim=256):
        super().__init__()
        self.embd_dim = embd_dim # 词嵌入维度
        self.hidden_size = hidden_size # 隐藏层维度
        self.input_size = input_size   # 输入词汇表大小 
        self.embedding = nn.Embedding(input_size, embd_dim)
        self.gru = nn.GRU(embd_dim, hidden_size,batch_first=True) # 词嵌入层， 隐藏层维度

    def forward(self, input, hidden):
        """前向传播"""
        embedded = self.embedding(input)  # [1,seq_len] -> [1, seq_len, hidden_size]
        output, hidden = self.gru(embedded, hidden)    # output: [1, seq_len, hidden_size], hidden: [1, 1, hidden_size]
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)

def test_encoder():
    """测试编码器"""
    input_size = english_words_count
    hidden_size = 256
    embd_dim = 256
    encoder = EncoderRNN(input_size, embd_dim, hidden_size).to(device)
    input_tensor = torch.tensor([english_words2index[word] for word in "i am a student".split(' ')] + [EOS_TOKEN], dtype=torch.long).to(device)
    hidden = encoder.init_hidden()
    output, hidden = encoder(input_tensor, hidden)
    print(f"Encoder output shape: {output.shape}, output:\n {output}")
    print(f"Encoder hidden shape: {hidden.shape}, hidden:\n {hidden}")

class DecoderRNN_No_Attention(nn.Module):
    """解码器RNN（无注意力机制）"""
    def __init__(self, hidden_size, output_size,embd_dim=256):
        super().__init__()
        self.embd_dim = embd_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embd_dim)
        self.drop = nn.Dropout(0.1)
        self.gru = nn.GRU(embd_dim, hidden_size,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self, input, hidden):
        """前向传播"""
        embedded = self.embedding(input) # [1, seq_len] -> [1, seq_len, hidden_size]
        embedded = self.drop(embedded)  # dropout随即失活10%
        output, hidden = self.gru(embedded, hidden)    # output: [1, seq_len, hidden_size], hidden: [1, 1, hidden_size]
        output = self.out(output[0])  # [1, seq_len, hidden_size] -> [1, output_size]
        return output, hidden
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)

def test_decoder_No_Attention():
    """测试解码器（无注意力机制）"""
    dataloader = my_dataloader()
    my_encoder = EncoderRNN(input_size=english_words_count, embd_dim=256, hidden_size=512).to(device)
    my_decoder = DecoderRNN_No_Attention(hidden_size=512, output_size=french_words_count, embd_dim=256).to(device)
    print(f"my_encoder: {my_encoder}")
    print(f"my_decoder: {my_decoder}")

    for i, (x, y) in enumerate(dataloader):
        print(f"Batch {i}: x shape: {x.shape}, y shape: {y.shape}")
        print(f"x: {x}, y: {y}")
        h0 = my_encoder.init_hidden()
        encoder_output, encoder_hidden = my_encoder(x, h0)
        decoder_hidden = encoder_hidden  # 解码器的初始隐藏状态为编码器
        for j in range(y.shape[1]):  # 遍历目标句子的每个词
            tmp = y[0][j].view(-1, 1)  # 将目标词转换为形状 [1, 1]
            decoder_output, decoder_hidden = my_decoder(tmp, decoder_hidden)
            print(f"第 {j + 1} 个词: decoder_output shape: {decoder_output.shape}, decoder_hidden shape: {decoder_hidden.shape}")
        if i == 0:  # 只测试第一个批次
            break

class Decoder_Attention(nn.Module):
    def __init__(self, hidden_size, output_size, embd_dim=256):
        super().__init__()
        self.embd_dim = embd_dim
        self.hidden_size = hidden_size   
        self.output_size = output_size   
        self.embedding = nn.Embedding(output_size, embd_dim)
        self.drop = nn.Dropout(0.1)
        self.attn_combine = nn.Linear(hidden_size + embd_dim, hidden_size)  # 注意力结合层
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)  # GRU层
        self.out = nn.Linear(hidden_size, output_size)  # 输出层
    def forward(self, input, hidden, encoder_outputs):
        """前向传播
        input: [batch_size, 1] 目标词索引
        hidden: [1, batch_size, hidden_size] 解码器的隐藏状态，当Q
        encoder_outputs: [batch_size, seq_len, hidden_size] 编码器的输出当成K和V
        """
        embedded = self.embedding(input)  # [1, seq_len] -> [1, seq_len, embd_dim]
        embedded = self.drop(embedded)  # dropout随即失活10%
        # 计算注意力权重
        score = torch.bmm(hidden.transpose(0, 1), encoder_outputs.transpose(-2, -1)) / math.sqrt(self.hidden_size)  
        # Q.transpose[batch_size,1,hidden_size] * K^T[batch_size, hidden_size, seq_len] -> [batch_size, 1, seq_len]
        attn_weights = F.softmax(score, dim=-1)  # [batch_size, 1, seq_len]
        # 计算上下文向量
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, seq_len] * [batch_size, seq_len, hidden_size] -> [batch_size, 1, hidden_size]
        combined = torch.cat((embedded, context), dim=-1)  # [batch_size, 1, embd_dim + hidden_size]
        combined = self.attn_combine(combined)  # [batch_size, 1, hidden_size]
        combined = F.relu(combined)  # 激活函数
        output, hidden = self.gru(combined, hidden)  # GRU层
        output = self.out(output[0])  # [batch_size, 1, hidden_size] -> [batch_size, output_size]
        return output, hidden, attn_weights
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size).to(device)

def test_decoder_Attention():
    """测试解码器（带注意力机制）"""
    dataloader = my_dataloader()
    my_encoder = EncoderRNN(input_size=english_words_count, embd_dim=256, hidden_size=512).to(device)
    my_decoder = Decoder_Attention(hidden_size=512, output_size=french_words_count, embd_dim=256).to(device)
    print(f"my_encoder: {my_encoder}")
    print(f"my_decoder: {my_decoder}")

    for i, (x, y) in enumerate(dataloader):
        print(f"Batch {i}: x shape: {x.shape}, y shape: {y.shape}")
        print(f"x: {x}, y: {y}")
        h0 = my_encoder.init_hidden()
        encoder_output, encoder_hidden = my_encoder(x, h0)
        decoder_hidden = encoder_hidden  # 解码器的初始隐藏状态为编码器
        for j in range(y.shape[1]):  # 遍历目标句子的每个词
            tmp = y[0][j].view(-1, 1)  # 将目标词转换为形状 [1, 1]
            decoder_output, decoder_hidden, attn_weights = my_decoder(tmp, decoder_hidden, encoder_output)
            print(f"第 {j + 1} 个词: decoder_output shape: {decoder_output.shape}, decoder_hidden shape: {decoder_hidden.shape}, attn_weights shape: {attn_weights.shape}")
        if i == 0:  # 只测试第一个批次
            break

my_lr,epoch,teacher_forcing_ratio,print_interval_num,plot_interval_num = 0.0001, 10, 0.5, 1000, 100

def one_train(encoder_rnn,decoder_rnn,encoder_optimizer,decoder_optimizer,criterion,x,y):
    """单次训练
    :param encoder_rnn: 编码器模型
    :param decoder_rnn: 解码器模型
    :param encoder_optimizer: 编码器优化器
    :param decoder_optimizer: 解码器优化器
    :param criterion: 损失函数
    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0.0
    h0 = encoder_rnn.init_hidden()
    encoder_output, encoder_hidden = encoder_rnn(x, h0) # 编码器输出和隐藏状态
    decoder_hidden = encoder_hidden # 解码器的初始隐藏状态为编码器的隐藏状态
    input_y = torch.tensor([[SOS_TOKEN]], dtype=torch.long).to(device) # 解码器的输入为起始符号
    teacher_forcing = True if random.random() < teacher_forcing_ratio else False # 是否使用教师强制
    if teacher_forcing: # 使用教师强制
        for i in range(y.shape[1]): # 遍历目标句子的每个词
            decoder_output, decoder_hidden, _ = decoder_rnn(input_y, decoder_hidden, encoder_output) # 解码器输出和隐藏状态
            loss += criterion(decoder_output, y[0][i].view(-1)) # 计算损失
            input_y = y[0][i].view(-1, 1) # 下一个输入为目标句子的下一个词
    else: # 不使用教师强制
        for i in range(y.shape[1]): # 遍历目标句子的每个词
            decoder_output, decoder_hidden, _ = decoder_rnn(input_y, decoder_hidden, encoder_output) # 解码器输出和隐藏状态
            loss += criterion(decoder_output, y[0][i].view(-1)) # 计算损失
            topv, topi = decoder_output.topk(1) # 获取预测的词索引
            if topi.item() == EOS_TOKEN: # 如果预测的词是结束符号，则停止解码
                break
            input_y = topi.detach() # 下一个输入为预测的词索引
    loss.backward() # 反向传播

    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(
    list(encoder_rnn.parameters()) + list(decoder_rnn.parameters()), 5.0)

    encoder_optimizer.step() # 更新编码器参数
    decoder_optimizer.step() # 更新解码器参数
    return loss.item() / y.shape[1] # 返回平均损失

def train():
    """训练模型"""
    dataloader = my_dataloader()
    encoder_rnn = EncoderRNN(input_size=english_words_count, embd_dim=256, hidden_size=512).to(device) # 编码器模型
    decoder_rnn = Decoder_Attention(hidden_size=512, output_size=french_words_count, embd_dim=256).to(device) # 解码器模型
    encoder_optimizer = optim.Adam(encoder_rnn.parameters(), lr=my_lr)
    decoder_optimizer = optim.Adam(decoder_rnn.parameters(), lr=my_lr)
    criterion = nn.CrossEntropyLoss()
    loss_list = []
    for e in range(epoch):
        print_total_loss = 0.0
        plot_total_loss = 0.0
        total_loss = 0.0
        start_time = time.time()
        for i, (x, y) in enumerate(dataloader):
            loss = one_train(encoder_rnn, decoder_rnn, encoder_optimizer, decoder_optimizer, criterion, x, y)
            print_total_loss += loss
            plot_total_loss += loss
            total_loss += loss
            if (i + 1) % print_interval_num == 0:
                print_avg_loss = print_total_loss / print_interval_num
                print(f"Epoch [{e + 1}/{epoch}], Step [{i + 1}/{len(dataloader)}], print_avg_loss: {print_avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")
                print_total_loss = 0.0
            if(i + 1) % plot_interval_num == 0:
                avg_loss = plot_total_loss / plot_interval_num
                loss_list.append(avg_loss)
                plot_total_loss = 0.0
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{e + 1}/{epoch}], Average Loss: {avg_loss:.4f}")
        torch.save(encoder_rnn.state_dict(), f"./model/encoder_rnn_{e + 1}.pth")
        torch.save(decoder_rnn.state_dict(), f"./model/decoder_rnn_{e + 1}.pth")
    # 绘制损失曲线
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Curve')
    plt.savefig('./img/loss_curve.png')
    plt.show()
    return loss_list

def evaluate(sentence):
    """评估模型，翻译输入句子"""
    epoch = 5
    encoder_rnn = EncoderRNN(input_size=english_words_count, embd_dim=256, hidden_size=512).to(device)
    decoder_rnn = Decoder_Attention(hidden_size=512, output_size=french_words_count, embd_dim=256).to(device)
    encoder_rnn.load_state_dict(torch.load(f"./model/encoder_rnn_{epoch}.pth"))
    decoder_rnn.load_state_dict(torch.load(f"./model/decoder_rnn_{epoch}.pth"))
    encoder_rnn.eval()
    decoder_rnn.eval()
    with torch.no_grad():
        words = normalize_string(sentence).split(' ')
        idx = [english_words2index[w] for w in words] + [EOS_TOKEN]
        h0 = encoder_rnn.init_hidden()
        input_tensor = torch.tensor(idx, dtype=torch.long).to(device).unsqueeze(0)  # [1, seq_len]
        encoder_output, encoder_hidden = encoder_rnn(input_tensor, h0)
        decoder_hidden = encoder_hidden
        input_y = torch.tensor([[SOS_TOKEN]], dtype=torch.long).to(device)
        translated_sentence = []
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, _ = decoder_rnn(input_y, decoder_hidden, encoder_output)
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_TOKEN:
                break
            translated_sentence.append(french_index2words[topi.item()])
            input_y = topi.detach()
        return ' '.join(translated_sentence)



if __name__ == "__main__":
    # test_decoder_No_Attention()
    # test_decoder_Attention()
    # train()
    # 测试翻译
    test_sentences = [
    " we re in love.",
    "i am very interested in french .",
]
    for sentence in test_sentences:
        translation = evaluate(sentence)
        print(f"English: {sentence} -> French: {translation}")