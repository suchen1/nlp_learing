import fasttext
import os
os.system('cls' if os.name == 'nt' else 'clear')   


def dm01_normal():
    model1 = fasttext.train_supervised(input="./data/cooking_train.txt")
    print(model1.test("./data/cooking_valid.txt"))
def dm02_preprocess():
    """对数据做预处理"""
    model2 = fasttext.train_supervised(input="./data/cooking.pre.train")
def dm03_epoch():
    model2 = fasttext.train_supervised(input="./data/cooking.pre.train", epoch=50)
def dm04_lr():
    model2 = fasttext.train_supervised(input="./data/cooking.pre.train", epoch = 50, lr = 0.3)
    print(model2.test("./data/cooking.pre.valid"))
def dm05_ngram_loss():
    model2 = fasttext.train_supervised(input="./data/cooking.pre.train", epoch = 50, lr = 1, wordNgrams=2, loss='hs')
    # hs: hierarchical softmax, 适合大类目分类(层次化softmax) : 负采样+softmax+哈夫曼树
    print(model2.test("./data/cooking.pre.valid"))
def dm06_auto():
    model2 = fasttext.train_supervised(input="./data/cooking.pre.train",
                                        autotuneValidationFile="./data/cooking.pre.valid", 
                                        autotuneDuration=600) # 自动调参，autotuneDuration: 训练时间，单位秒
def dm07_multi_label():
    # ova: one-vs-all, 适合多标签分类
    model2 = fasttext.train_supervised(input="./data/cooking.pre.train", epoch = 50, lr = 0.3, wordNgrams=2, loss='ova')
    print(model2.predict("How to make a cake ?", k=3, threshold=0.1)) # 返回前  k个预测结果
def dm08_save():
    model2 = fasttext.train_supervised(input="./data/cooking.pre.train", epoch = 50, lr = 0.3, wordNgrams=2, loss='ova')
    model2.save_model("model.bin") # 保存模型
    model3 = fasttext.load_model("./model/model.bin") # 加载模型
    print(model3.predict("How to make a cake ?", k=3, threshold=0.1)) # 返回前  k个预测结果
if __name__ == "__main__":
    # dm01_normal() # (3000, 0.14966666666666667, 0.06472538561337754)
    # dm02_preprocess() # (3000, 0.15366666666666667, 0.06645524001729855)
    # dm03_epoch() # (3000, 0.5566666666666666, 0.24073807121233962)
    # dm04_lr() # (3000, 0.5633333333333334, 0.2436211618855413)
    # dm05_ngram_loss() # (3000, 0.597, 0.25818076978520976)
    # dm06_auto() # (3000, 0.597, 0.25818076978520976)
    # dm07_multi_label() # (3000, 0.6155555555556, 0.321231412321)
    dm08_save() # 
