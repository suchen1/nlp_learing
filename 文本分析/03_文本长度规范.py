import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import jieba
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.preprocessing.sequence as seq

def padding():
    # 1. 准备文本数据
    texts = [
        "我爱自然语言处理",
        "深度学习是人工智能的核心",
        "词嵌入可以将词语映射到向量空间，我很喜欢词嵌入",
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

    # 4. 对序列进行填充
    padded_sequences = seq.pad_sequences(sequences, maxlen=10, padding='pre', truncating='post')
    return padded_sequences

if __name__ == '__main__':
    print("填充后的序列:", padding())