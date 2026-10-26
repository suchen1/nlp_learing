import os
os.system('cls' if os.name == 'nt' else 'clear')     # 先清屏再跑
from transformers import pipeline # 管道
import numpy as np
from rich import print

# 模型目录（和上面各 test 里的 "./model/xxx" 同一个基准：工作目录 = 迁移学习/）
MODEL_DIR = "./model"

"""
    transformers库主要有三个用法：
    1. 使用pipeline快速使用预训练模型   管道
    2. 使用AutoModel和AutoTokenizer加载预训练模型  自动模型
    3. 使用Trainer进行模型训练  具体模型

"""

def test_sentiment_classification():
    # 情感分析
    classifier = pipeline(task = "sentiment-analysis", model = "./model/chinese_sentiment")
    result = classifier("我讨厌你！！")
    print(result)

def test_feature_extraction():
    # 特征提取
    feature_extractor = pipeline(task = "feature-extraction", model = "./model/bert-base-chinese")
    result = feature_extractor("我讨厌你！！")
    print(np.array(result).shape) # (1, 8, 768)  1个句子，8个词（包含起始和结束），768维向量（12个头 * 64维向量）

def test_fill_mask():
    # 填充掩码
    # ★ 必须用"带 MLM 头"的模型：
    #   bert-base-chinese 是 BertForMaskedLM（有 mlm 头）-> 能正常填空
    #   chinese-bert-wwm  是 BertModel（只有编码器，无 mlm 头）-> 头是随机的，输出繁体乱码
    fill_mask = pipeline(task = "fill-mask", model = "./model/chinese-bert-wwm")
    result = fill_mask("你好[MASK]心啊！！") # top_k表示返回前几个最可能的结果
    print(result)

def test_question_answering():
    # 问答
    # ★ context 必须是"一段材料"，不能是口语自我介绍：
    #   这个模型在 MRC（机器阅读理解）数据上微调，[CLS] 专门表示"无答案"。
    #   给它口语短句，答案虽然蒙对了，但 score 会≈0（模型认为"这段里抽不出答案"）。
    question_answerer = pipeline(task = "question-answering",
                                 model = "./model/chinese_pretrain_mrc_roberta_wwm_ext_large")
    context = ("吴昊，1995年出生于浙江杭州，2018年毕业于浙江大学计算机系，"
               "现为某互联网公司算法工程师，主要研究方向是自然语言处理和大模型微调。")
    print("  材料:", context)
    for q in ["吴昊是哪一年出生的？", "他毕业于哪所大学？", "他的研究方向是什么？"]:
        r = question_answerer(question=q, context=context)
        print("    %-22s -> 「%s」  score=%.4f" % (q, r["answer"], r["score"]))

    print()
    print("  （反例）给口语短句，score 趋近 0：")
    r = question_answerer(question="你是谁？", context="我是wuhao,我爱学习大模型，我喜欢游戏。")
    print("    %-22s -> 「%s」  score=%.6f" % ("你是谁？", r["answer"], r["score"]))
def test_summeration():
    contest = "Throughout history and across the world, those who achieve great things all hold firmly to their original intentions and keep their promises. " \
    "In the past, Ji Bu valued his word like gold, and Wei Sheng kept his vow to hug the pillar; " \
    "today, seeing you switch sides easily on the battlefield, abandoning the southwest to join Zheng," \
    "isn't that just like Lord Ye who only liked dragons in name? Confucius said: 'If a person has no trustworthiness, " \
    "I don't know what they can do.' Since you fear difficulty and change your mind, how could simple sparrows understand the ambition of a swan? " \
    "I hope you reflect on yourself thrice, and don't let your classmates laugh at you as a cowardly 'Zheng drifter.'"

    my_model = pipeline(task = "summarization", model = "./model/distilbart-cnn-12-6")
    print(my_model(contest))

def test_ner():
    # 命名实体识别
    my_model = pipeline(task = "ner", model = "./model/roberta-base-finetuned-cluener2020-chinese")
    text = "马云在杭州创立了阿里巴巴，李四和王五都在北京工作。"
    print("原文:", text)
    result = my_model(text)
    print("识别结果:", result)
if __name__ == "__main__":
    # test_sentiment_classification() # [{'label': 'star 1', 'score': 0.5189510583877563}]
    # test_feature_extraction() # (1, 8, 768)
    # test_fill_mask() 
    # test_question_answering() # {'score': 0.9999999403953552, 'start': 0, 'end': 7, 'answer': '我是一个AI助手。'}
    # test_summeration() 
    test_ner()