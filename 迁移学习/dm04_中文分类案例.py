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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MyTokenizer = BertTokenizer.from_pretrained('./model/bert-base-chinese')

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

def collate_fn1(data):
    """处理批量数据，对data统一格式并转成模型输入
    :param data: 批量数据(list)"""
    sents = [item['text'] for item in data]
    label = [item['label'] for item in data]
    input = MyTokenizer(sents, padding=True, truncation=True, return_tensors='pt').to(device)
    return input, torch.LongTensor(label).to(device)


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
        self.fc = nn.Linear(768, 2)  # 有两个类别

        for p in self.bert.parameters():
            p.requires_grad = False  # 冻结BERT参数
    def train(self, mode=True):
        super().train(mode)
        self.bert.eval()                       # 冻结了就一直 eval
        return self
    def forward(self, input):
        with torch.no_grad():  # 不建计算图，省显存（参数已靠 requires_grad=False 冻结）
            output = self.bert(**input)
            pooled_output = output.pooler_output  # 获取池化输出
        logits = self.fc(pooled_output)  # 全连接层输出
        return logits

def test():
    train_dataset, test_dataset, validation_dataset = file2dataset()
    for input,label in get_dataloader(train_dataset, batch_size=32, shuffle=False):
        input.to(device)
        label.to(device)
        output = AiModel().to(device)(input)
        print(f'[bold green]output: {output}')
        print(f'[bold red]预测的label: {output.argmax(dim=1)}')
        print(f'[bold blue]label: {label}')
        break

def train(epochs = 10):
    train_dataset, test_dataset, validation_dataset = file2dataset()
    mydataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    model = AiModel().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for i, (input, label) in enumerate(tqdm(mydataloader)):
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if (i + 1) % 20 == 0:
        
                print(f'[bold green]Epoch {epoch + 1}/{epochs}, Step {i + 1}/{len(mydataloader)}, Loss: {loss.item():.4f}[/bold green]')
        if (epoch + 1) % 5 == 0:        
            torch.save(model.state_dict(), f'./model/classification_{epoch+1}.pth')

def evaluate():
    train_dataset, test_dataset, validation_dataset = file2dataset()
    dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)
    Model = AiModel().to(device)
    Model.load_state_dict(torch.load('./model/classification_10.pth'))
    Model.eval()
    correct_predictions = 0
    total = 0
    with torch.no_grad():
        for i,(input, label) in enumerate(dataloader):
            output = Model(input)
            predictions = output.argmax(dim=1)
            correct_predictions += (predictions == label).sum().item()
            total += label.size(0)
            avg_acc = correct_predictions / total
            print(f'[bold green]Step {i + 1}/{len(dataloader)}, Accuracy: {avg_acc:.4f}[/bold green]')


if __name__ == '__main__':
    # train_dataset, test_dataset, validation_dataset = file2dataset()
    # test()
    # train(epochs=10)
    evaluate()