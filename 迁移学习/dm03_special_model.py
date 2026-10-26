import torch
from transformers import BertTokenizer, BertModel, BertConfig #比AutoModel更底层的类，能更灵活地控制模型结构
from transformers import BertForSequenceClassification #比AutoModelForSequenceClassification更底层的类，能更灵活地控制模型结构
from transformers import BertForMaskedLM #比AutoModelForMaskedLM更底层的类
from rich import print


def test_sentiment_classification():
    """情感分析"""
    Mytokenizer = BertTokenizer.from_pretrained("./model/chinese_sentiment")
    MyModel = BertForSequenceClassification.from_pretrained("./model/chinese_sentiment")
    MyModel.eval()                                  # 推理模式：关掉 dropout，结果才稳定
    inputs = Mytokenizer("我讨厌你！！", return_tensors="pt")
    with torch.no_grad():                           # 只做推理，不需要梯度
        outputs = MyModel(**inputs)

    # outputs.logits 是"未归一化的分数(logits)"，形状 (batch_size, num_labels) = (1, 5)
    # ★ torch 的 Tensor 没有 marginalize 方法（marginalize 是 pyro 里的概念，指把某一维边缘化掉），
    #   想要"softmax 后的概率分布"，直接用 torch.softmax 即可
    probs = torch.softmax(outputs.logits, dim=-1)   # 等价于 dim=1，因为这里只有第 0 维 batch、第 1 维类别
    print(probs)                                    # tensor([[p0, p1, p2, p3, p4]])，和为 1

    # 配合模型自带的 id2label，把"第几类"翻译成"几星"
    id2label = MyModel.config.id2label
    for i, p in enumerate(probs[0]):
        print(f"{id2label[i]:<8} {p.item():.6f}")

if __name__ == "__main__":
    test_sentiment_classification()