from math import e
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # 禁用 oneDNN 优化器，避免报错
os.system('cls' if os.name == 'nt' else 'clear')     # 先清屏再跑
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from datasets import DatasetDict, load_dataset  # huggingface数据集加载器
from typing import cast
import time
from rich import print
from tqdm import tqdm
from torch.optim import AdamW                 # 优化器
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MyTokenizer = BertTokenizer.from_pretrained('./model/bert-base-chinese')
SPECIAL = {MyTokenizer.cls_token_id, MyTokenizer.sep_token_id, MyTokenizer.pad_token_id}

def file2dataset():
    ds = cast(DatasetDict, load_dataset('csv', data_files={'train': './data/train.csv',
                                                           'test': './data/test.csv',
                                                           'validation': './data/validation.csv'}))
    train_dataset = ds['train']
    test_dataset = ds['test']
    validation_dataset = ds['validation']
    print(f'[bold green]训练集样本数: {len(train_dataset)}[/bold green]')
    print(f'[bold blue]测试集样本数: {len(test_dataset)}[/bold blue]')
    print(f'[bold yellow]验证集样本数: {len(validation_dataset)}[/bold yellow]')
    return train_dataset, test_dataset, validation_dataset

def collate_fn1(data, max_length=32):
    sents = [item['text'] for item in data]
    enc = MyTokenizer(sents, padding=True, truncation=True,
                      max_length=max_length, return_tensors='pt')
    # print(f'[bold blue]修改前的input_ids: {enc["input_ids"]}[/bold blue]')
    lengths = enc['attention_mask'].sum(dim=1)        # 每条真实长度
    min_len = int(lengths.min())
    if min_len <= 2:                                  # 太短就没得遮
        rand = 1
    else:
        rand = random.randint(1, min_len - 2)         # 位置 1 ~ min_len-2（跳过 CLS/SEP）
    # print(f'[bold green]随机选择的索引: {rand}[/bold green]')
    labels = enc['input_ids'][:, rand].clone()  # 取出对应位置的真实词 id
    enc['input_ids'][:, rand] = MyTokenizer.mask_token_id
    # print(f'[bold yellow]修改后的input_ids: {enc["input_ids"]}[/bold yellow]')
    return enc.to(device), labels.to(device), rand

def get_dataloader(dataset, batch_size=32, shuffle=True):
    """获取数据加载器
    :param dataset: 数据集
    :param batch_size: 批量大小
    :param shuffle: 是否打乱数据
    :return: 数据加载器"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn1)


class AiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('./model/bert-base-chinese')
        self.fc = nn.Linear(768, MyTokenizer.vocab_size)  # 输出维度改为词表大小

        for p in self.bert.parameters():
            p.requires_grad = False  # 冻结BERT参数
    def train(self, mode=True):
        super().train(mode)
        self.bert.eval()                       
        return self
    def forward(self, input, rand=None):
        with torch.no_grad():  # 不建计算图，省显存（参数已靠 requires_grad=False 冻结）
            output = self.bert(**input)
        logits = self.fc(output.last_hidden_state[:,rand])  # 输出维度改为词表大小
        return logits

def train(epochs = 5,batch_size = 32):
    train_dataset, test_dataset, validation_dataset = file2dataset()
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    model = AiModel().to(device)
    model.load_state_dict(torch.load('./model/MLM_10.pth', map_location=device))  # 加载模型参数
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    model.train()
    total = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for i,(input, labels, rand) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            logits = model(input, rand)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 20 == 0:
                # ★ 最近 20 步的平均 —— 看得清趋势
                print(f'[bold green]Epoch {epoch+1}/{epochs}, '
                      f'Step {i+1}/{len(train_dataloader)}, '
                      f'最近20步平均 loss: {total_loss/20:.4f}[/bold green]')
                total_loss = 0.0        # ★ 清零，下一个 20 步重新算
        print(f'[bold blue]===== Epoch {epoch+1} 结束 =====[/bold blue]')
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'./model/MLM_{epoch+1}.pth')

def show_inputs(input_ids):
    """把 [MASK] 显示成 ★，其他 special token 丢掉"""
    out = []
    for row in input_ids.tolist():
        toks = []
        for x in row:
            if x == MyTokenizer.mask_token_id:
                toks.append('★')                       # [MASK] -> ★
            elif x in (MyTokenizer.cls_token_id,
                       MyTokenizer.sep_token_id,
                       MyTokenizer.pad_token_id):
                continue                               # 其他特殊 token 丢掉
            else:
                toks.append(MyTokenizer.convert_ids_to_tokens([x])[0])
        out.append(''.join(toks))
    return out

def evaluate():
    train_dataset, test_dataset, validation_dataset = file2dataset()
    dataloader = get_dataloader(test_dataset, batch_size=12, shuffle=False)
    Model = AiModel().to(device)
    Model.load_state_dict(torch.load('./model/MLM_10.pth'))
    Model.eval()
    correct_predictions = 0
    total = 0
    with torch.no_grad():
       for i,(input, labels, rand) in enumerate(dataloader):
            # print(f'input.shape: {input["input_ids"].shape}, labels.shape: {labels.shape}')
            output = Model(input,rand)
            # print(f'output.shape: {output.shape}, labels.shape: {labels.shape}')
            predictions = output.argmax(dim=-1)
            # print(f'predictions.shape: {predictions.shape}, labels.shape: {labels.shape}')
            # break  # 只看第一批，避免输出过多
            correct_predictions += (predictions == labels).sum().item()
            total += labels.size(0)
            avg_acc = correct_predictions / total
            if (i + 1) % 10 == 0:
                print(f'[bold green]Step {i + 1}/{len(dataloader)}, Accuracy: {avg_acc:.4f}[/bold green]')
                print(f'[bold blue]输入: {show_inputs(input["input_ids"])}[/bold blue]')
                # ② 真值 vs 预测，逐条并排
                for j in range(len(labels)):
                    true_word = MyTokenizer.convert_ids_to_tokens([labels[j].item()])[0]
                    pred_word = MyTokenizer.convert_ids_to_tokens([predictions[j].item()])[0]
                    mark = '✓' if true_word == pred_word else '✗'
                    print(f'  [{j}] ★位置={rand}  真值「{true_word}」  预测「{pred_word}」  {mark}')

if __name__ == '__main__':
    # train(epochs=10, batch_size=32)
    evaluate()
