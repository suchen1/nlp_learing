import os
os.system('cls' if os.name == 'nt' else 'clear')     # 先清屏再跑
from transformers import pipeline # 管道
import numpy as np
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




if __name__ == "__main__":
    # test_sentiment_classification() # [{'label': 'star 1', 'score': 0.5189510583877563}]
    test_feature_extraction() # [[[[-0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549, -0.05999999865889549, 0.05999999865889549]]]]
