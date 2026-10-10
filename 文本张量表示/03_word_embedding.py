import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import jieba
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

def dm01_word_embedding():
    # 1. 准备文本数据
    texts = [
        "我爱自然语言处理",
        "深度学习是人工智能的核心",
        "词嵌入可以将词语映射到向量空间",
        "机器学习和深度学习是人工智能的重要分支"
    ]

    # 2. 分词
    words = [list(jieba.lcut(text)) for text in texts]
    print("分词结果:", words)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words)
    print(tokenizer.word_index)

    # 3. 将文本转换为序列
    sequences = tokenizer.texts_to_sequences(words)
    print("文本序列:", sequences)

    # 4. 将序列转换为张量
    embd = nn.Embedding(num_embeddings=len(tokenizer.word_index)+1, embedding_dim=8)
    my_summary = SummaryWriter(log_dir=r'D:\ML_Project\nlp\runs')
    my_summary.add_embedding(embd.weight,
                             metadata=['<PAD>'] + list(tokenizer.word_index.keys()),
                             tag='word_embedding')
    my_summary.close()

if __name__ == "__main__":
    dm01_word_embedding()