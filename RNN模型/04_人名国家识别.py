import torch
import torch.nn as nn
import torch.nn. functional as F
import torch. optim as optim                          # 优化器
from torch.utils. data import Dataset, DataLoader     # 数据集和数据加载器
import string
import time
import matplotlib. pyplot as plt
from tqdm import tqdm                                # 进度条
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import glob, json, os
#获取常用字符包括字母和常用标点
all_letters = string.ascii_letters + " .,;'"
#获取常用字符数量
n_letters = len(all_letters)
categorys = ['Italian', 'English', 'Arabic','Spanish','Scottish', 'Irish','Chinese', 'Vietnamese',
'Japanese','French','Greek','Dutch','Korean','Polish','Portuguese','Russian','Czech','German']
#国家名个数
categorynum = len(categorys)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_data(file_path):
    my_list_x, my_list_y = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            my_list_x.append(line[0])
            my_list_y.append(line[1])

    categorynum = len(set(my_list_y))
    print(f'样本个数：{len(my_list_x)}')
    print(f'国家名种类数：{len(set(my_list_y))}')
    return my_list_x, my_list_y

class My_ClassDataset(Dataset):
    def __init__(self, my_list_x, my_list_y):
        self.my_list_x = my_list_x
        self.my_list_y = my_list_y
        self.len = len(self.my_list_x) # 样本总数
        self.char2idx = {c: i for i, c in enumerate(all_letters)}
        self.cat2idx = {c: i for i, c in enumerate(categorys)}

    def __len__(self):
        return len(self.my_list_x)

    def __getitem__(self, idx):
        idx = max(0, min(idx, self.len - 1))  # 确保索引在有效范围内
        x = self.my_list_x[idx]
        y = self.my_list_y[idx]

        # 对x one-hot编码
        x_tensor = torch.zeros(len(x), n_letters)  # 创建一个全零张量，形状为 (len(x), n_letters)
        for i, char in enumerate(x):
            if char in all_letters:
                x_tensor[i][self.char2idx[char]] = 1  # 将对应位置设为1
        y_tensor = torch.tensor(self.cat2idx[y], dtype=torch.long)  # 将y转换为索引张量
        return x_tensor, y_tensor

def my_Dataloader(batch_size):
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    dataset = My_ClassDataset(my_list_x, my_list_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader  # x.shape = (batch_size, seq_len, n_letters), y.shape = (batch_size,)

class My_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,n_laysers=1):
        super().__init__()
        self.input_size = input_size # 输入维度
        self.hidden_size = hidden_size # 性能（隐藏层维度）
        self.output_size = output_size # 输出维度（18个国家）
        self.n_layers = n_laysers
        self.rnn = nn.RNN(input_size, hidden_size, n_laysers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden=None):
        out, hn = self.rnn(x, hidden)
        return self.fc(hn[-1])  # 只取最后一层的隐藏状态作为输出

    def inithidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size))

class My_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,n_laysers=1):
        super().__init__()
        self.input_size = input_size # 输入维度
        self.hidden_size = hidden_size # 性能（隐藏层维度）
        self.output_size = output_size # 输出维度（18个国家）
        self.n_layers = n_laysers
        self.rnn = nn.LSTM(input_size, hidden_size, n_laysers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden,cell):
        out, (hn,cn) = self.rnn(x, (hidden, cell))
        return self.fc(hn[-1])  # 只取最后一层的隐藏状态作为输出

    def inithidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                torch.zeros(self.n_layers, batch_size, self.hidden_size))

class My_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,n_laysers=1):
        super().__init__()
        self.input_size = input_size # 输入维度
        self.hidden_size = hidden_size # 性能（隐藏层维度）
        self.output_size = output_size # 输出维度（18个国家）
        self.n_layers = n_laysers
        self.rnn = nn.GRU(input_size, hidden_size, n_laysers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        out, hn = self.rnn(x, hidden)
        return self.fc(hn[-1])  # 只取最后一层的隐藏状态作为输出

    def inithidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size))

lr, epochs = 0.001,5

def train_rnn():
    dataloader = my_Dataloader(batch_size=1)
    model = My_RNN(input_size=n_letters, hidden_size=128, output_size=categorynum).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time() # 模型开始训练时间.
    total_loss = 0.0 # 已训练的损失和
    total_loss_list = [] # 每100个样本求一次平均损失，形成:损失列表.
    total_acc_num = 0 # 已训练的成本，预测准确总数
    total_acc_list = [] # 每100个样本求一次平均准确率，形成:准确率列表。
    seen = 0
    for epoch in range(epochs):
        for i, (x, y) in enumerate(tqdm(dataloader)):
            seen += 1
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            hidden = model.inithidden(batch_size=1).to(device)
            output = model(x, hidden)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            total_acc_num += (pred == y).sum().item()
            if (i + 1) % 100 == 0:
                total_loss_list.append(total_loss / seen)
                total_acc_list.append(total_acc_num / seen)

            if (i + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], '
                      f'Loss: {total_loss / (i + 1):.4f}, Accuracy: {total_acc_num / (i + 1):.4f},'
                      f'Time: {time.time() - start_time:.2f}s')

    torch.save(model.state_dict(), './model/my_rnn.pth')
    elapsed = time.time() - start_time
    save_train_log('rnn', total_loss_list, total_acc_list, elapsed,
                   log_dir='./log', epochs=epochs, lr=lr, batch_size=1)
    return total_loss_list, elapsed, total_acc_list

def train_lstm():
    dataloader = my_Dataloader(batch_size=1)
    model = My_LSTM(input_size=n_letters, hidden_size=128, output_size=categorynum).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time() # 模型开始训练时间.
    total_loss = 0.0 # 已训练的损失和
    total_loss_list = [] # 每100个样本求一次平均损失，形成:损失列表.
    total_acc_num = 0 # 已训练的成本，预测准确总数
    total_acc_list = [] # 每100个样本求一次平均准确率，形成:准确率列表。
    seen = 0
    for epoch in range(epochs):
        for i, (x, y) in enumerate(tqdm(dataloader)):
            seen += 1
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            hidden, cell = [t.to(device) for t in model.inithidden(batch_size=1)]
            output = model(x, hidden,cell)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            total_acc_num += (pred == y).sum().item()
            if (i + 1) % 100 == 0:
                total_loss_list.append(total_loss / seen)
                total_acc_list.append(total_acc_num / seen)

            if (i + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], '
                      f'Loss: {total_loss / (i + 1):.4f}, Accuracy: {total_acc_num / (i + 1):.4f},'
                      f'Time: {time.time() - start_time:.2f}s')

    torch.save(model.state_dict(), './model/my_lstm.pth')
    elapsed = time.time() - start_time
    save_train_log('lstm', total_loss_list, total_acc_list, elapsed,
                   log_dir='./log', epochs=epochs, lr=lr, batch_size=1)
    return total_loss_list, elapsed, total_acc_list

def train_gru():
    dataloader = my_Dataloader(batch_size=1)
    model = My_GRU(input_size=n_letters, hidden_size=128, output_size=categorynum).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time() # 模型开始训练时间.
    total_loss = 0.0 # 已训练的损失和
    total_loss_list = [] # 每100个样本求一次平均损失，形成:损失列表.
    total_acc_num = 0 # 已训练的成本，预测准确总数
    total_acc_list = [] # 每100个样本求一次平均准确率，形成:准确率列表。
    seen = 0
    for epoch in range(epochs):
        for i, (x, y) in enumerate(tqdm(dataloader)):
            seen += 1
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            hidden = model.inithidden(batch_size=1).to(device)
            output = model(x, hidden)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            total_acc_num += (pred == y).sum().item()
            if (i + 1) % 100 == 0:
                total_loss_list.append(total_loss / seen)
                total_acc_list.append(total_acc_num / seen)

            if (i + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], '
                      f'Loss: {total_loss / seen:.4f}, Accuracy: {total_acc_num / seen:.4f},'
                      f'Time: {time.time() - start_time:.2f}s')

    torch.save(model.state_dict(), './model/my_gru.pth')
    elapsed = time.time() - start_time
    save_train_log('gru', total_loss_list, total_acc_list, elapsed,
                   log_dir='./log', epochs=epochs, lr=lr, batch_size=1)
    return total_loss_list, elapsed, total_acc_list

def save_train_log(model_name, loss_list, acc_list, elapsed, log_dir='./log', **extra):
    """把一个模型的训练过程写成 JSON：一个模型一个文件，方便后面画图对比。"""
    os.makedirs(log_dir, exist_ok=True)
    data = {
        'model': model_name,                                  # 'rnn' / 'lstm' / 'gru'
        'elapsed_seconds': round(float(elapsed), 2),
        'points': len(loss_list),
        'loss_list': [round(float(v), 6) for v in loss_list],
        'acc_list':  [round(float(v), 6) for v in acc_list],
    }
    data.update(extra)                                        # epochs / lr / batch_size 等超参数
    path = os.path.join(log_dir, f'{model_name}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'[保存] {path}（{data["points"]} 个点，耗时 {data["elapsed_seconds"]}s）')
    return path

def plot_train_logs(log_dir='./log', save_dir=None):
    """读 log_dir 下所有 json，画三张图：损失对比 / 准确率对比 / 耗时对比。"""
    files = sorted(glob.glob(os.path.join(log_dir, '*.json')))
    if not files:
        print('没找到日志文件:', log_dir)
        return
    runs = [json.load(open(f, encoding='utf-8')) for f in files]
    save_dir = save_dir or log_dir
    os.makedirs(save_dir, exist_ok=True)

    # 图 1：损失下降
    plt.figure(figsize=(9, 4.5))
    for r in runs:
        plt.plot(range(1, len(r['loss_list']) + 1), r['loss_list'], label=r['model'], linewidth=1.2)
    plt.xlabel('记录点（每 100 个样本一个点）'); plt.ylabel('平均损失')
    plt.title('训练损失对比'); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'loss_compare.png'), dpi=120); plt.close()

    # 图 2：准确率上升
    plt.figure(figsize=(9, 4.5))
    for r in runs:
        plt.plot(range(1, len(r['acc_list']) + 1), r['acc_list'], label=r['model'], linewidth=1.2)
    plt.xlabel('记录点（每 100 个样本一个点）'); plt.ylabel('平均准确率')
    plt.title('训练准确率对比'); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'acc_compare.png'), dpi=120); plt.close()

    # 图 3：耗时对比
    plt.figure(figsize=(6, 4))
    names = [r['model'] for r in runs]
    times = [r['elapsed_seconds'] for r in runs]
    bars = plt.bar(names, times)
    for b, t in zip(bars, times):
        plt.text(b.get_x() + b.get_width() / 2, t, f'{t:.1f}s', ha='center', va='bottom')
    plt.ylabel('训练总耗时（秒）'); plt.title('三个模型训练耗时对比'); plt.grid(axis='y', alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'time_compare.png'), dpi=120); plt.close()

    print('三张图已保存到:', save_dir)

def name2tensor(name):
    """把名字转换为 one-hot 编码的张量，形状为 (1, len(name), n_letters)"""
    tensor = torch.zeros(len(name), n_letters)
    for i, char in enumerate(name):
        if char in all_letters:
            tensor[i][all_letters.index(char)] = 1
    return tensor.unsqueeze(0)

def predict(name):
    my_rnn = My_RNN(input_size=n_letters, hidden_size=128, output_size=categorynum).to(device)
    my_rnn.load_state_dict(torch.load('./model/my_rnn.pth', map_location=device))
    my_lstm = My_LSTM(input_size=n_letters, hidden_size=128, output_size=categorynum).to(device)
    my_lstm.load_state_dict(torch.load('./model/my_lstm.pth', map_location=device))
    my_gru = My_GRU(input_size=n_letters, hidden_size=128, output_size=categorynum).to(device)
    my_gru.load_state_dict(torch.load('./model/my_gru.pth', map_location=device))

    tensor_name = name2tensor(name).to(device)
    hidden_rnn = my_rnn.inithidden(batch_size=1).to(device)
    hidden_lstm, cell_lstm = [t.to(device) for t in my_lstm.inithidden(batch_size=1)]
    hidden_gru = my_gru.inithidden(batch_size=1).to(device)

    with torch.no_grad():
        output_rnn = my_rnn(tensor_name, hidden_rnn)
        output_lstm = my_lstm(tensor_name, hidden_lstm, cell_lstm)
        output_gru = my_gru(tensor_name, hidden_gru)
        print(f'带预测的名字: {name}')
        rnn_topv, rnn_topi = torch.topk(F.softmax(output_rnn, dim=1), 3)
        print(f'RNN Top3预测结果: {[categorys[i.item()] for i in rnn_topi[0]]}, 概率: {[round(v.item(), 4) for v in rnn_topv[0]]}')
        lstm_topv, lstm_topi = torch.topk(F.softmax(output_lstm, dim=1), 3)
        print(f'LSTM Top3预测结果: {[categorys[i.item()] for i in lstm_topi[0]]}, 概率: {[round(v.item(), 4) for v in lstm_topv[0]]}')
        gru_topv, gru_topi = torch.topk(F.softmax(output_gru, dim=1), 3)
        print(f'GRU Top3预测结果: {[categorys[i.item()] for i in gru_topi[0]]}, 概率: {[round(v.item(), 4) for v in gru_topv[0]]}')

if __name__ == '__main__':
    predict('Wang')

