"""
 CBOW模型和Skip-gram模型的实现
    CBOW:从两边到中间的推理
    Skip-gram:从中间到两边的推理
"""

import fasttext

def dm01_CBOW():
    model = fasttext.train_unsupervised(input='./data/ai20ag', model='cbow', epoch=10,lr =0.1, dim=100, ws=5, minCount=1)
    model.save_model('./model/CBOW_model.bin')
    print("CBOW模型训练完成并保存为CBOW_model.bin")


def dm02_SkipGram():
    model = fasttext.train_unsupervised(input='./data/ai20ag', model='skipgram', epoch=10,lr =0.1, dim=100, ws=5, minCount=1)
    model.save_model('./model/SkipGram_model.bin')
    print("Skip-gram模型训练完成并保存为SkipGram_model.bin")

def dm03_load_model():
    model = fasttext.load_model('./model/SkipGram_model.bin')
    print("加载Skip-gram模型完成")
    print(model.get_word_vector("money"))
    print(model.get_nearest_neighbors("dog"))
if __name__ == '__main__':
    # dm01_CBOW()
    # dm02_SkipGram()
    dm03_load_model()