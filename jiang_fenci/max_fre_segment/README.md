# 基于动态规划的最大频率分词

## 写在前面

- 基于动态规划的最大频率分词是一种比较简单的分词方法，jieba分词就用了这种分词方法（当然还有HMM）
- 博主对该分词方法做了一下简单的测试

## 测试数据集

- 选的是微软的分词测试集和清华的分词测试集
- 两个测试集都有一定的训练数据和测试数据
- 基本情况如下：

|数据集|训练集数量|测试集数量|
|----|----|------|
|微软|86924|3985|
|清华|19056|1944|

## 测试方法

- 既然选择了最大频率方法，当然首先是同级最大频率
- 博主选了jieba的词典和hanlp的词典以及上述两个数据集的训练集生成的词典，情况如下

|词典集|测试集|PRE|RECAL|F1|
|----|----|----|----|----|
|微软|微软|0.841|0.913|0.875|
|微软|清华|0.783|0.853|0.817|
|清华|微软|0.776|0.868|0.819|
|清华|清华|0.794|0.888|0.838|
|jieba|微软|0.801|0.871|0.835|
|jieba|清华|0.826|0.865|0845|
|hanlp|微软|0.654|0.786|0.714|
|hanlp|清华|0.698|0.807|0.748|

## 一点儿结论

- 还是自己的训练集测试自己的数据好！
- 核心词典的话jieba的挺好用的，具有很好的通用性（也是因为最大频率就是jieba用的算法之一）
- hanlp模块很多，词典的权重没那么高！
- 微软的数据更加的特立独行一点儿， 具体表现在自己的训练集格外高，其他的训练集都是不如清华的
- 维特比算法从头开始和从尾巴开始没有区别（待证明，自己没证明出来）

## 写在后面

- NLP任重道远！但是冲鸭！
- 欢迎关注Github：https://github.com/chuanfanyoudong/nlp_learn