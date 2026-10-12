#导包
import jieba #分词包
import seaborn as sns #画图包
import pandas as pd #数据处理包
import matplotlib.pyplot as plt #画图包,推荐 pip install matplotlib=3.9
from itertools import chain #迭代器工具
import jieba.posseg as pseg #词性标注包，例如:动词，名词，形容词...
from wordcloud import WordCloud #词云包
#解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
#用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

def dm01_label_sns_countplot():

    plt.style.use('fivethirtyeight')

    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    test_data = pd.read_csv('./data/dev.tsv',sep='\t')
    print(train_data.head())
    sns.countplot(x='label', data=train_data,hue = 'label')
    plt.tight_layout()
    plt.show()

    sns.countplot(x='label', data=test_data, hue='label')
    plt.tight_layout()
    plt.show()

def dm02_len_sns_displot():
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    test_data = pd.read_csv('./data/dev.tsv', sep='\t')

    train_data['text_len'] = train_data['sentence'].apply(lambda x: len(x))
    test_data['text_len'] = test_data['sentence'].apply(lambda x: len(x))

    sns.countplot(x='text_len', data=train_data)
    plt.xticks([i for i in range(0, 300, 20)])
    plt.tight_layout()
    plt.show()

    sns.histplot(x='text_len', data=train_data,kde=True)
    plt.show()

def dm03_len_sns_stripplot():
    plt.style.use('fivethirtyeight')
    plt.tight_layout()
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    test_data = pd.read_csv('./data/dev.tsv', sep='\t')

    train_data['text_len'] = train_data['sentence'].apply(lambda x: len(x))
    test_data['text_len'] = test_data['sentence'].apply(lambda x: len(x))
    sns.stripplot(x='label', y='text_len', data=train_data,hue='label')
    plt.show()

def dm04_wordcount():
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    test_data = pd.read_csv('./data/dev.tsv', sep='\t')

    train_data['text_len'] = train_data['sentence'].apply(lambda x: len(x))
    test_data['text_len'] = test_data['sentence'].apply(lambda x: len(x))

    # 统计词频
    all_words = set(chain(*train_data['sentence'].apply(lambda x: jieba.lcut(x)).tolist()))
    print("所有词的个数:", len(all_words))
    print("所有词:", all_words)

def get_a_list(text):
    a_list = []
    for i in pseg.lcut(text):
        if i.flag == 'a':
            a_list.append(i.word)
    return a_list

def get_wordcloud(key_words):
    wordcloud = WordCloud(font_path='simhei.ttf',
                          max_words=300,
                          background_color='white')
    key_words_str = ' '.join(key_words)
    wordcloud.generate(key_words_str)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def dm05_wordcloud():
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    test_data = pd.read_csv('./data/dev.tsv', sep='\t')
    """
    train_data_0 = train_data[train_data['label'] == 0]
        key_words_train_0 = []
        for text in train_data_0['sentence']:
            key_words_train_0.extend(get_a_list(text))
        get_wordcloud(key_words_train_0)
    """
    train_data_0 = train_data[train_data['label'] == 0]['sentence']
    key_words_train_0 =  set(chain(*map(lambda x: get_a_list(x), train_data_0)))
    get_wordcloud(key_words_train_0)

if __name__ == '__main__':
    # dm01_label_sns_countplot()
    # dm02_len_sns_displot()
    # dm03_len_sns_stripplot()
    # dm04_wordcount()
    # get_a_list('价格确实比较高，而且还没有早餐提供。')
    dm05_wordcloud()