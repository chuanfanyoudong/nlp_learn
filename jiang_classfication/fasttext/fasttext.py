import time

import os
from fastText import train_supervised
import sys
from conf.config import get_config
__config = get_config()
DATA_PATH = __config["classfication"]["data_path"]
ROOT = __config["path"]["root"]
sys.path.append(ROOT)
DATA_PATH = ROOT + DATA_PATH
#
#
# def get_data(file_name):
#     f = open(DATA_PATH + file_name)
#     write_file = open(DATA_PATH + "fast_text.test", "w", encoding= "utf-8")
#     num = 0
#     label_dict = {}
#     label_num = 1
#     corpus_list = []
#     label_list = []
#     for line in f:
#         # print(line)
#         num += 1
#         line = line.strip()
#         label = line[:2]
#         if label not in label_dict:
#             label_dict[label] = label_num
#             label_num += 1
#         content = line[3:]
#         # if type in data_dict:
#         list = jieba.lcut(content)
#         # print(list)
#         if list != []:
#             write_file.write("__label__" + str(label_dict[label]) + " , ")
#             write_file.write(" ".join(list) + "\n")
#             label_list.append(label)
#             corpus_list.append(" ".join(list))
#     # print(corpus_list, label_list)
#     return corpus_list, label_list
#
# # get_data("cnews.test.txt")


def print_results(N, p, r):
    ### 输出格式配置，配置数据数量，准确率和召回率
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

if __name__ == "__main__":
    ## 配置输入数据的路径
    train_data = DATA_PATH + "fast_text.train"
    ## 配置测试文件的路径
    valid_data = DATA_PATH + "fast_text.test"
    ## 说一下参数的意思：
    ## input:训练文件路径（按照格式定义好）
    ## epoch: 迭代次数
    ## lr：学习率
    ## wordNgrams：窗口大小
    ## loss：计算loss的方法，主要有softmax和hs（霍夫曼树）
    start = time.time()
    model = train_supervised(
        input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1,
        loss="hs"
    )
    end = time.time()
    print("训练{}模型需要的时间是{}".format("fasttext", end - start))
    ## 打印在测试集上的结果
    print_results(*model.test(valid_data))
    ##存储模型
    model.save_model("cooking.bin")
    ## 模型优化，就是减少模型的内存占用
    model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
    ## 看看优化内存占用后的模型的效果
    print_results(*model.test(valid_data))
    ## 再来存储一波
    model.save_model("cooking.ftz")
