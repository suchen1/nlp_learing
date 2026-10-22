import math
import re
import os
os.system('cls' if os.name == 'nt' else 'clear')     # 先清屏再跑
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence   # ★ pack/unpack
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

# ★ PAD（补齐符）：加在词表【末尾】，不打乱已有编号，所以 SOS=0 / EOS=1 / 真实词从 2 开始全都不变
#   英文词表 0~2802 → PAD 用 2803；法文词表 0~4344 → PAD 用 4345
#   对应的 Embedding 要开 count+1 行（多出一行专给 PAD）
PAD_TOKEN_EN = english_words_count    # 2803
PAD_TOKEN_FR = french_words_count     # 4345


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
        # ★ 改成返回【列表】：长度各不相同，交给 collate_fn 去补齐
        #   （原来 .to(device) 放在这里也不好：dataset 里常驻 GPU 张量，还挡不住"长度不一"）
        return english_indices, french_indices


def my_collate(batch):
    """★ 补齐（pad）：把长度不一的句子补成 (B, L) 的矩形，否则 DataLoader 堆不起来

    batch 形如 [[[3,5,7,1], [9,2,1]], [[3,5,1], [9,2,4,1]], ...]   每句长度不同
    返回：en_pad (B, L_en)  fr_pad (B, L_fr)  en_len (B,)  fr_len (B,)
    """
    ens, frs = zip(*batch)                                   # 拆成两个 tuple of list
    B = len(ens)
    en_max = max(len(s) for s in ens)                         # 这个 batch 里最长的英文句
    fr_max = max(len(s) for s in frs)

    # ★ 长度张量必须在【CPU】上、且是 int64 —— pack_padded_sequence 的硬性要求
    en_len = torch.tensor([len(s) for s in ens], dtype=torch.long)
    fr_len = torch.tensor([len(s) for s in frs], dtype=torch.long)

    en_pad = torch.full((B, en_max), PAD_TOKEN_EN, dtype=torch.long)   # ① 整张先填 PAD
    fr_pad = torch.full((B, fr_max), PAD_TOKEN_FR, dtype=torch.long)
    for i, s in enumerate(ens):
        en_pad[i, :len(s)] = torch.tensor(s, dtype=torch.long)          # ② 把真词盖上去
    for i, s in enumerate(frs):
        fr_pad[i, :len(s)] = torch.tensor(s, dtype=torch.long)

    return en_pad.to(device), fr_pad.to(device), en_len, fr_len


def my_dataloader(batch_size=1, shuffle=True):
    """创建数据加载器（★ 传 collate_fn，否则组不了 batch）"""
    dataset = My_Dataset(my_pairs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=my_collate)

class EncoderRNN(nn.Module):
    """编码器RNN"""
    def __init__(self, input_size, hidden_size, embd_dim=256, pad_idx=None):
        super().__init__()
        self.embd_dim = embd_dim # 词嵌入维度
        self.hidden_size = hidden_size # 隐藏层维度
        self.input_size = input_size   # 输入词汇表大小
        # ★ padding_idx=PAD：那一行的词向量固定为 0，且永远拿不到梯度（PAD 不该被"学到"）
        self.embedding = nn.Embedding(input_size, embd_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embd_dim, hidden_size,batch_first=True) # 词嵌入层， 隐藏层维度

    def forward(self, input, lengths):
        """前向传播
        input:   (B, L)  已补齐的编号（短的句子后面全是 PAD）
        lengths: (B,)    ★ 每句的真实长度 —— pack 就靠它知道"读到哪停"
        返回 output (B, L, H) 和 hidden (1, B, H)
        """
        embedded = self.embedding(input)                        # (B, L) -> (B, L, E)

        # ★① pack：把 (B,L,E) 压成"只含真实词"的形态，GRU 读到真实长度就停
        #    enforce_sorted=False → 不用自己把 batch 按长度降序排（PyTorch 内部排完会还原）
        packed = pack_padded_sequence(embedded, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        out_packed, hidden = self.gru(packed)                   # hidden: (1, B, H) ★ 已还原成原始顺序

        # ★② unpack：变回 (B, L, H) 的整齐张量（PAD 位置填 0），给解码器当 K/V
        #    total_length 固定成输入长度，避免不同 batch 形状飘
        output, _ = pad_packed_sequence(out_packed, batch_first=True,
                                       total_length=input.size(1))
        return output, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

def test_encoder():
    """测试编码器（★ 已适配 batch 接口：要传 lengths）"""
    input_size = english_words_count + 1          # ★ 多一行给 PAD
    hidden_size = 512
    embd_dim = 256
    encoder = EncoderRNN(input_size, hidden_size, embd_dim, pad_idx=PAD_TOKEN_EN).to(device)
    idx = [english_words2index[word] for word in "i am a student".split(' ')] + [EOS_TOKEN]
    input_tensor = torch.tensor(idx, dtype=torch.long).to(device).unsqueeze(0)   # (1, L)
    lengths = torch.tensor([len(idx)], dtype=torch.long)                        # ★ (1,)
    output, hidden = encoder(input_tensor, lengths)
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
    """测试解码器（无注意力机制）★ 已适配 batch 接口"""
    dataloader = my_dataloader(batch_size=4)          # 用 4 句一个小 batch 演示补齐
    my_encoder = EncoderRNN(input_size=english_words_count + 1, hidden_size=512,
                            embd_dim=256, pad_idx=PAD_TOKEN_EN).to(device)
    my_decoder = DecoderRNN_No_Attention(hidden_size=512, output_size=french_words_count + 1,
                                         embd_dim=256).to(device)
    print(f"my_encoder: {my_encoder}")
    print(f"my_decoder: {my_decoder}")

    for i, (x, y, x_len, y_len) in enumerate(dataloader):
        print(f"Batch {i}: x {x.shape}, y {y.shape}, x_len {x_len.tolist()}")
        print(f"x: {x}")
        encoder_output, encoder_hidden = my_encoder(x, x_len)          # ★ 传 lengths
        decoder_hidden = encoder_hidden
        for j in range(y.shape[1]):
            tmp = y[:, j].unsqueeze(1)                                 # ★ (B,) -> (B,1)
            decoder_output, decoder_hidden = my_decoder(tmp, decoder_hidden)
            print(f"第 {j+1} 个词: decoder_output {decoder_output.shape}, hidden {decoder_hidden.shape}")
        if i == 0:  # 只测试第一个批次
            break

class Decoder_Attention(nn.Module):
    def __init__(self, hidden_size, output_size, embd_dim=256, pad_idx=None):
        super().__init__()
        self.embd_dim = embd_dim
        self.hidden_size = hidden_size   
        self.output_size = output_size   
        self.embedding = nn.Embedding(output_size, embd_dim, padding_idx=pad_idx)  # ★ PAD 行不学
        self.drop = nn.Dropout(0.1)
        self.attn_combine = nn.Linear(hidden_size + embd_dim, hidden_size)  # 注意力结合层
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)  # GRU层
        self.out = nn.Linear(hidden_size, output_size)  # 输出层
    def forward(self, input, hidden, encoder_outputs, enc_mask=None):
        """前向传播
        input: [batch_size, 1] 目标词索引
        hidden: [1, batch_size, hidden_size] 解码器的隐藏状态，当Q
        encoder_outputs: [batch_size, seq_len, hidden_size] 编码器的输出当成K和V
        enc_mask: [batch_size, seq_len] bool，True=真实词  False=PAD  ★ 新增
        """
        embedded = self.embedding(input)  # [1, seq_len] -> [1, seq_len, embd_dim]
        embedded = self.drop(embedded)  # dropout随即失活10%
        # 计算注意力权重
        score = torch.bmm(hidden.transpose(0, 1), encoder_outputs.transpose(-2, -1)) / math.sqrt(self.hidden_size)  
        # Q.transpose[batch_size,1,hidden_size] * K^T[batch_size, hidden_size, seq_len] -> [batch_size, 1, seq_len]
        # ★ mask：把 PAD 位置的分数打成 -inf，softmax 之后它们的权重就精确为 0
        #   注意两个细节：① 要取反 `~`（masked_fill 填的是 True 的位置）
        #                ② unsqueeze(1) 把 (B,L) 扩成 (B,1,L) 才能和 score 广播
        if enc_mask is not None:
            score = score.masked_fill(~enc_mask.unsqueeze(1), float('-inf'))
        attn_weights = F.softmax(score, dim=-1)  # [batch_size, 1, seq_len]
        # 计算上下文向量
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, seq_len] * [batch_size, seq_len, hidden_size] -> [batch_size, 1, hidden_size]
        combined = torch.cat((embedded, context), dim=-1)  # [batch_size, 1, embd_dim + hidden_size]
        combined = self.attn_combine(combined)  # [batch_size, 1, hidden_size]
        combined = F.relu(combined)  # 激活函数
        output, hidden = self.gru(combined, hidden)  # GRU层
        # ★ 这里原来是 self.out(output[0])：batch=1 没问题，但 batch>1 时 output[0] 只取【第 0 个样本】
        #   → 形状静默变成 (1, V)。改成 squeeze(1) 把 (B,1,H) 压成 (B,H)，整个 batch 都保留
        output = self.out(output.squeeze(1))  # [batch_size, hidden_size] -> [batch_size, output_size]
        return output, hidden, attn_weights
    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

def test_decoder_Attention():
    """测试解码器（带注意力机制）★ 已适配 batch 接口"""
    dataloader = my_dataloader(batch_size=4)          # 4 句一个 batch，能看出 mask 的效果
    my_encoder = EncoderRNN(input_size=english_words_count + 1, hidden_size=512,
                            embd_dim=256, pad_idx=PAD_TOKEN_EN).to(device)
    my_decoder = Decoder_Attention(hidden_size=512, output_size=french_words_count + 1,
                                   embd_dim=256, pad_idx=PAD_TOKEN_FR).to(device)
    print(f"my_encoder: {my_encoder}")
    print(f"my_decoder: {my_decoder}")

    for i, (x, y, x_len, y_len) in enumerate(dataloader):
        print(f"Batch {i}: x {x.shape}, y {y.shape}, x_len {x_len.tolist()}")
        print(f"x: {x}")
        enc_mask = (x != PAD_TOKEN_EN)                        # ★ (B, L) True=真词
        encoder_output, encoder_hidden = my_encoder(x, x_len)
        decoder_hidden = encoder_hidden
        for j in range(y.shape[1]):
            tmp = y[:, j].unsqueeze(1)                        # ★ (B,) -> (B,1)
            decoder_output, decoder_hidden, attn_weights = my_decoder(
                tmp, decoder_hidden, encoder_output, enc_mask)  # ★ 传 mask
            print(f"第 {j+1} 个词: out {decoder_output.shape}, hidden {decoder_hidden.shape}, "
                  f"attn {attn_weights.shape}")
        print(f"  第 0 句的注意力（真实 {int(x_len[0])} 词，PAD 位应为 0）: "
              f"{[round(v, 3) for v in attn_weights[0, 0].tolist()]}")
        if i == 0:  # 只测试第一个批次
            break

my_lr,epoch,teacher_forcing_ratio,print_interval_num,plot_interval_num = 0.0001, 10, 0.5, 100, 50
BATCH_SIZE = 32          # ★ 一次喂 32 个句子（原来是 1）

def one_train(encoder_rnn,decoder_rnn,encoder_optimizer,decoder_optimizer,criterion,x,y,x_len):
    """单次训练（★ 支持 batch：一次算 B 个句子）

    :param x:      (B, L_en) 英文编号，已补齐
    :param y:      (B, L_fr) 法文编号，已补齐
    :param x_len:  (B,)      英文真实长度
    :param criterion: 必须是 CrossEntropyLoss(ignore_index=PAD_TOKEN_FR)
    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    B = x.size(0)
    # ★ 源句的 mask：True=真实词，False=PAD。解码器用它把 PAD 的注意力权重压成 0
    enc_mask = (x != PAD_TOKEN_EN)

    encoder_output, encoder_hidden = encoder_rnn(x, x_len)  # 编码器：pack 吃掉 PAD
    decoder_hidden = encoder_hidden                          # 中间语义当解码器初始状态
    # ★ 解码器第一步的输入：B 行、每行都是 SOS
    input_y = torch.full((B, 1), SOS_TOKEN, dtype=torch.long, device=device)

    loss = 0.0
    teacher_forcing = random.random() < teacher_forcing_ratio
    if teacher_forcing:  # 使用教师强制
        for i in range(y.shape[1]):  # 遍历【时间步】（batch 维度天然并行）
            decoder_output, decoder_hidden, _ = decoder_rnn(input_y, decoder_hidden, encoder_output, enc_mask)
            loss += criterion(decoder_output, y[:, i])   # ★ (B,V) vs (B,) 一次算整个 batch
            input_y = y[:, i].unsqueeze(1)               # ★ (B,) -> (B,1)，下一步喂【真值】
    else:  # 不使用教师强制
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for i in range(y.shape[1]):
            decoder_output, decoder_hidden, _ = decoder_rnn(input_y, decoder_hidden, encoder_output, enc_mask)
            loss += criterion(decoder_output, y[:, i])
            topi = decoder_output.argmax(1)              # ★ (B,) 每句各自的预测，不再是 .item()
            finished |= (topi == EOS_TOKEN)              # 记录谁已经生成了 EOS
            input_y = topi.unsqueeze(1)                  # 喂自己的预测
            if finished.all():                           # 整个 batch 都结束了才停
                break

    loss = loss / y.shape[1]  # ★ 先取平均再 backward（和 return 的口径一致）
    loss.backward()  # 反向传播

    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(
    list(encoder_rnn.parameters()) + list(decoder_rnn.parameters()), 5.0)

    encoder_optimizer.step() # 更新编码器参数
    decoder_optimizer.step() # 更新解码器参数
    return loss.item()  # 已经是平均损失

def train(total_epochs=None, resume_from=0):
    """训练模型

    :param total_epochs: 总共要训到第几轮（默认用全局 epoch）
    :param resume_from:  0 = 从头训；
                         10 = 加载 model/*_10.pth 接着训，从第 11 轮开始存
    """
    total_epochs = epoch if total_epochs is None else total_epochs
    dataloader = my_dataloader(batch_size=BATCH_SIZE)                                     # ★ batch=32
    # ★ input_size 要 +1：词表末尾多出一行给 PAD；pad_idx 让那行不参与梯度
    encoder_rnn = EncoderRNN(input_size=english_words_count + 1, embd_dim=256,
                             hidden_size=512, pad_idx=PAD_TOKEN_EN).to(device)
    decoder_rnn = Decoder_Attention(hidden_size=512, output_size=french_words_count + 1,
                                    embd_dim=256, pad_idx=PAD_TOKEN_FR).to(device)
    encoder_optimizer = optim.Adam(encoder_rnn.parameters(), lr=my_lr)
    decoder_optimizer = optim.Adam(decoder_rnn.parameters(), lr=my_lr)

    # ★ 断点续训：把已有权重灌进【同一个】模型对象，然后接着循环
    if resume_from > 0:
        enc_path = f"./model/encoder_rnn_{resume_from}.pth"
        dec_path = f"./model/decoder_rnn_{resume_from}.pth"
        encoder_rnn.load_state_dict(torch.load(enc_path, map_location=device))
        decoder_rnn.load_state_dict(torch.load(dec_path, map_location=device))
        print(f"★ 已加载第 {resume_from} 轮权重 → 从第 {resume_from + 1} 轮继续"
              f"（目标第 {total_epochs} 轮）")
    else:
        print(f"★ 从头训练，共 {total_epochs} 轮")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_FR)                            # ★ 跳过 PAD
    loss_list, epoch_losses = [], []                                                      # ★ 两种粒度分开存
    for e in range(total_epochs - resume_from):        # ★ 只跑剩下的轮数
        real_epoch = resume_from + e + 1               # ★ 真实轮次编号（用于打印和存档）
        print_total_loss = 0.0
        plot_total_loss = 0.0
        total_loss = 0.0
        start_time = time.time()
        for i, (x, y, x_len, fr_len) in enumerate(dataloader):                            # ★ 多拿一个长度
            loss = one_train(encoder_rnn, decoder_rnn, encoder_optimizer, decoder_optimizer,
                             criterion, x, y, x_len)
            print_total_loss += loss
            plot_total_loss += loss
            total_loss += loss
            if (i + 1) % print_interval_num == 0:
                print(f"Epoch [{real_epoch}/{total_epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"loss: {print_total_loss / print_interval_num:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")
                print_total_loss = 0.0
            if (i + 1) % plot_interval_num == 0:
                loss_list.append(plot_total_loss / plot_interval_num)                     # 细粒度：每 N 步
                plot_total_loss = 0.0
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)                                                     # 粗粒度：每轮（★ 别塞进 loss_list）
        print(f"Epoch [{real_epoch}/{total_epochs}], Average Loss: {avg_loss:.4f}")
        torch.save(encoder_rnn.state_dict(), f"./model/encoder_rnn_{real_epoch}.pth")
        torch.save(decoder_rnn.state_dict(), f"./model/decoder_rnn_{real_epoch}.pth")
    # 绘制损失曲线：细粒度/粗粒度分开画
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.xlabel(f'Step (x{plot_interval_num})')
    plt.ylabel('Average Loss')
    plt.title('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Loss per Epoch')
    plt.tight_layout()
    plt.savefig('./img/loss_curve.png')
    plt.show()
    return loss_list

def evaluate(sentence, epoch=50):
    """评估模型，翻译输入句子（单句推理：batch=1，没有 PAD，所以 mask 可以不传）

    :param epoch: 用 model/ 里第几轮的权重（当前只保留了第 50 轮；
                  以后重训到别的轮次，记得改这个默认值）
    """
    encoder_rnn = EncoderRNN(input_size=english_words_count + 1, embd_dim=256,
                             hidden_size=512, pad_idx=PAD_TOKEN_EN).to(device)
    decoder_rnn = Decoder_Attention(hidden_size=512, output_size=french_words_count + 1,
                                    embd_dim=256, pad_idx=PAD_TOKEN_FR).to(device)
    encoder_rnn.load_state_dict(torch.load(f"./model/encoder_rnn_{epoch}.pth", map_location=device))
    decoder_rnn.load_state_dict(torch.load(f"./model/decoder_rnn_{epoch}.pth", map_location=device))
    encoder_rnn.eval()
    decoder_rnn.eval()
    with torch.no_grad():
        words = normalize_string(sentence).split(' ')
        missing = [w for w in words if w not in english_words2index]      # ★ OOV 保护
        if missing:
            return f'（跳过）词表里没有: {missing}'
        idx = [english_words2index[w] for w in words] + [EOS_TOKEN]
        input_tensor = torch.tensor(idx, dtype=torch.long).to(device).unsqueeze(0)  # [1, seq_len]
        lengths = torch.tensor([len(idx)], dtype=torch.long)                        # ★ 单句长度 (1,)
        encoder_output, encoder_hidden = encoder_rnn(input_tensor, lengths)         # ★ 传 lengths 而不是 h0
        decoder_hidden = encoder_hidden
        input_y = torch.tensor([[SOS_TOKEN]], dtype=torch.long).to(device)
        translated_sentence = []
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, _ = decoder_rnn(input_y, decoder_hidden, encoder_output)
            topi = decoder_output.topk(1)[1]        # (1,1)
            if topi.item() == EOS_TOKEN:
                break
            translated_sentence.append(french_index2words[topi.item()])
            input_y = topi
        return ' '.join(translated_sentence)



if __name__ == "__main__":
    # test_encoder()
    # test_decoder_No_Attention()
    # test_decoder_Attention()
     train(50,10)  # ★ 从第 11 轮开始训到第 30 轮       
    #  test_sentences = ["we re in love .", "i am hungry .", "i am a student ."]
    #  for sentence in test_sentences:
    #      print(f"English: {sentence} -> French: {evaluate(sentence,10)}")