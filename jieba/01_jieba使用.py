import jieba

# 使用精确模式分词
seg_list = jieba.lcut("我来到电子科技大学，学的是软件工程", cut_all=False)
print("精确模式: " + "/ ".join(seg_list))  # 精确模式: 我/ 来到/ 电子科技/ 大学/ ，/ 学/ 的/ 是/ 软件工程
# 使用全模式分词
seg_list = jieba.lcut("我来到电子科技大学，学的是软件工程", cut_all=True)
print("全模式: " + "/ ".join(seg_list))  # 全模式: 我/ 来到/ 电子/ 电子科/ 电子科技/ 科技/ 大学/ ，/ 学/ 的/ 是/ 软件/ 软件工程/ 工程
# 使用搜索引擎模式分词
jieba.load_userdict("./data/use.txt")
seg_list = jieba.lcut_for_search("我来到电子科技大学，学的是软件工程")
print("搜索引擎模式: " + "/ ".join(seg_list))  # 搜索引擎模式: 我/ 来到/ 电子/ 科技/ 大学/ 电子科/ 电子科技大学/ ，/ 学/ 的/ 是/ 软件/ 工程/ 软件工程
seg_list = jieba.lcut("我来到电子科技大学，学的是软件工程", cut_all=False)
print("精确模式: " + "/ ".join(seg_list))  # 精确模式: 我/ 来到/ 电子科技/ 大学/ ，/ 学/ 的/ 是/ 软件工程