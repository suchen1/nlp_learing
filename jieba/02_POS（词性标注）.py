
"""
NER:Named Entity Recognition，命名实体识别
    用于识别文本中的命名实体，如人名、地名、组织机构名等。理解语义最小单元
POS：Part-of-Speech，词性标注
    标识出每个词的词性，如名词、动词、形容词等。理解语法结构
"""

import jieba.posseg as pseg
import jieba
contest = "我来到电子科技大学，学的是软件工程"
result = pseg.lcut(contest)
for word, flag in result:
    print(f"单词{word}，词性{flag}")
print('@'*59)
jieba.load_userdict("./data/use.txt")
result = pseg.lcut(contest)
for word, flag in result:
    print(f"单词{word}，词性{flag}")