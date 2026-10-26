import os
os.system('cls' if os.name == 'nt' else 'clear')     # 先清屏再跑
import torch
from transformers import AutoModel,AutoConfig,AutoTokenizer
# 用于文本分类、填充掩码、问答等任务的预训练模型
from transformers import AutoModelForSequenceClassification,AutoModelForMaskedLM,AutoModelForQuestionAnswering
# 用于特征提取的预训练模型
from transformers import AutoModelForSeq2SeqLM,AutoModelForTokenClassification
from rich import print
import numpy as np
def test_sentiment_classification():
    """情感分析"""
    Mytokenizer = AutoTokenizer.from_pretrained("./model/chinese_sentiment")
    MyModel = AutoModelForSequenceClassification.from_pretrained("./model/chinese_sentiment")
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
    print(f"预测类别: {id2label[int(probs.argmax(dim=-1))]}  "
          f"概率: {probs.max().item():.6f}")

def test_feature_extraction():
    """特征提取"""
    Mytokenizer = AutoTokenizer.from_pretrained("./model/bert-base-chinese")
    MyModel = AutoModel.from_pretrained("./model/bert-base-chinese")
    message = ["我讨厌你！！", "我爱你！！"]
    meg_tokens = Mytokenizer.encode_plus(message, return_tensors="pt", padding=True, truncation=True)
    print(f'meg_tokens: \n{meg_tokens}')
    print(f'最后一个隐藏层的形状: {MyModel(**meg_tokens).last_hidden_state.shape}') # (batch_size, seq_len, hidden_size)

def test_fill_mask():
    """填充掩码"""
    Mytokenizer = AutoTokenizer.from_pretrained("./model/chinese-bert-wwm")
    MyModel = AutoModelForMaskedLM.from_pretrained("./model/chinese-bert-wwm")
    contest = "余东阳是[MASK]逼吗？"
    MyModel.eval()                                  # 推理模式：关掉 dropout，结果才稳定
    input = Mytokenizer(contest, return_tensors="pt")
    print('input: \n', input)
    with torch.no_grad():
        outputs = MyModel(**input)
        # outputs.logits 是"未归一化的分数(logits)"，形状 (batch_size, seq_len, vocab_size)
        print(f'outputs.logits: \n{outputs.logits}')
        print(f'outputs.logits.shape: {outputs.logits.shape}') # (1, 8, 21128)  1个句子，8个词（包含起始和结束），21128个词表
        mask_predict_idx = torch.argmax(outputs.logits[0][5]).item()
        print(f'原句为：{contest}')
        print(f'概率最高的词为：{Mytokenizer.decode(mask_predict_idx)}，概率为：{torch.softmax(outputs.logits[0][5], dim=-1)[mask_predict_idx].item():.6f}')

def test_question_answering():
    """问答"""
    Mytokenizer = AutoTokenizer.from_pretrained("./model/chinese_pretrain_mrc_roberta_wwm_ext_large")
    MyModel = AutoModelForQuestionAnswering.from_pretrained("./model/chinese_pretrain_mrc_roberta_wwm_ext_large")
    MyModel.eval()
    context = "那些成就伟大的人都坚定地坚持最初的心愿，信守诺言。\
        过去，季布把信用看得像黄金一样。\
        可今天，余东阳在战场上轻易换阵，放弃西南交大去投靠郑大，失去勇气就不是能不能考上的问题了。\
        孔子说过：‘人不讲信用，我真不知道他还能做什么。’所以我是觉得，余东阳是个不讲信用的人。"
    questions = ["余东阳放弃了哪个学校？","余东阳投靠了哪个学校？", "余东阳能考上吗？"]
    print(f"上下文：\n {context}")
    for question in questions:
        input = Mytokenizer(question, context, return_tensors="pt")
        outputs = MyModel(**input)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = Mytokenizer.convert_tokens_to_string(Mytokenizer.convert_ids_to_tokens(input["input_ids"][0][answer_start:answer_end]))
        print(f"问题：{question}")
        if(question == '余东阳能考上吗？'):
            print(f"预测答案：不 能")
        else:
            print(f"预测答案：{answer}")

def test_summeration():
    """文本摘要"""
    Mytokenizer = AutoTokenizer.from_pretrained("./model/distilbart-cnn-12-6")
    MyModel = AutoModelForSeq2SeqLM.from_pretrained("./model/distilbart-cnn-12-6")
    contest = "Throughout history and across the world, those who achieve great things all hold firmly to their original intentions and keep their promises. " \
        "In the past, Ji Bu valued his word like gold, and Wei Sheng kept his vow to hug the pillar; " \
        "today, seeing you switch sides easily on the battlefield, abandoning the southwest to join Zheng," \
        "isn't that just like Lord Ye who only liked dragons in name? Confucius said: 'If a person has no trustworthiness, " \
        "I don't know what they can do.' Since you fear difficulty and change your mind, how could simple sparrows understand the ambition of a swan? " \
        "I hope you reflect on yourself thrice, and don't let your classmates laugh at you as a cowardly 'Zheng drifter.'"
    input = Mytokenizer(contest, return_tensors="pt")
    print(f'input: \n{input}')
    outputs = MyModel.generate(**input)
    print(f'outputs: \n{outputs}')
    print(f'预测内容:\n{Mytokenizer.decode(outputs[0], skip_special_tokens=True)}')

def test_ner():
    """命名实体识别"""
    Mytokenizer = AutoTokenizer.from_pretrained("./model/roberta-base-finetuned-cluener2020-chinese")
    MyModel = AutoModelForTokenClassification.from_pretrained("./model/roberta-base-finetuned-cluener2020-chinese")
    config = AutoConfig.from_pretrained("./model/roberta-base-finetuned-cluener2020-chinese")
    text = "马云在杭州创立了阿里巴巴，李四和王五都在北京工作。"
    input = Mytokenizer(text, return_tensors="pt")
    print(f'input: \n{input}')
    outputs = MyModel(**input)
    print(f'outputs: \n{outputs.logits.shape}') # (1, 26, 32)  1个句子，26个词（包含起始和结束），32个类别
    outputs = outputs.logits[0]                      # (L, 32)  取掉 batch 维
    tokens = Mytokenizer.convert_ids_to_tokens(input["input_ids"][0].tolist())
    print(f'tokens: \n{tokens}')
    SKIP = {"[CLS]", "[SEP]", "[PAD]"}
    assert len(tokens) == outputs.shape[0], "tokens 和 logits 长度不一致！"
    for i in range(outputs.shape[0]):                # L = token 个数
        token = tokens[i]                            # ★ 用 tokens[i]，不能用 text[i]
        if token in SKIP:                            # ★ in（判断是否属于集合），不是 ==
            continue
        score = torch.argmax(outputs[i])             # (32,) -> 最大分数的下标
        prob = torch.softmax(outputs[i], dim=-1)[score].item()
        # 注意：i 是【token 下标】，不是字符下标（中间夹着 [CLS]、数字/英文可能多对一）
        print(f'第{i:>2}个token：{token:<6} 预测类别：{config.id2label[score.item()]:<14} 概率：{prob:.6f}')
if __name__ == "__main__":
    # test_sentiment_classification() 
    # test_feature_extraction()
    # test_fill_mask()
    # test_question_answering()
    # test_summeration()
    test_ner()
