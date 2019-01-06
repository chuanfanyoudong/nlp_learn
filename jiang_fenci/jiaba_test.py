import jieba.posseg as pseg

words = pseg.cut("我爱中国")
print(list(words))