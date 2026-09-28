import jieba
from tensorflow.keras.preprocessing.text import Tokenizer # 用于文本分词和向量化
import joblib
# todo: 获取文本的one-hot编码
def dm01():
    vobs = ["我", "来到", "电子科技大学", "学", "的", "是", "软件工程"]
    tokenizer = Tokenizer() # 创建一个Tokenizer对象，最多保留100个词
    tokenizer.fit_on_texts(vobs)
    # 将文本转换为one-hot编码
    print(tokenizer.word_index)

    joblib.dump(tokenizer, './model/tokenizer.pkl')

# todo: 加载one-hot编码
def dm02():
    tokenizer = joblib.load('./model/tokenizer.pkl')
    vobs = ["我", "来到", "电子科技大学", "学", "的", "是", "软件工程"]
    for vob in vobs:
        one_hot = [0] * (len(tokenizer.word_index))  # 创建一个长度为词汇表大小的全零向量
        index = tokenizer.word_index[vob]
        one_hot[index - 1] = 1
        print(f"{vob}：{one_hot} \n")

if __name__ == "__main__":
    dm01()
    dm02()