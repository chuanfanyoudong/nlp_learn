#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com/chuanfanyoudong
@software: PyCharm
@file: hand_lstm.py
@time: 2019/3/19 10:23
"""

"""
参考网上的手写lstm
大佬链接：https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/（原文链接）
         https://blog.csdn.net/weixin_38776853/article/details/80156244
"""


# 导入相关库
import copy, numpy as np
# 设置随机种子
np.random.seed(0)
# 定义好sigmoid函数
def sigmoid(x):
    # 下面就是sigmoid的公式
    output = 1/(1 + np.exp(-x))
    # 返回结果
    return output

# 下面写出sigmoid的导数函数
def sigmoid_output_to_derivative(output):
    # sigmoid函数的导入就等于函数本身*（1- 函数）
    return output * (1-output)
# 将数值转化成二进制数字
int2binary = {}
# 声明二进制的位数8
binary_dim = 8
# 所以能存储的最大值为2的8次方
largest_number = pow(2, binary_dim)
# 下面这个unpackbits函数可以把整数转化成二进制数
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
# 把整数和二进制数的对应关系存储进来
for i in range(largest_number):
    int2binary[i] = binary[i]
# 学习步长速率
alpha = 0.1
# 输入维度
input_dim = 2
# 隐层维度
hidden_dim = 16
# 输出层维度
output_dim = 1
# np.random.random生成从0到1之间随机浮点数，2x-1使其取值范围在[-1, 1]。
# 初始化三套权重
synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1
# 声明三个更新权重
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)
# 开始训练数据啦
for j in range(10000):
    # 在这里初始化一个a + b = c的问题
    # 随机初始化一个a
    a_int = np.random.randint(largest_number / 2)  # int version
    # 将a转化成二进制
    a = int2binary[a_int]  # binary encoding
    # 随机初始化一个b
    b_int = np.random.randint(largest_number / 2)  # int version
    # 将b转化成二进制
    b = int2binary[b_int]  # binary encoding
    # 算出c c = a + b
    c_int = a_int + b_int
    # c转化成二进制
    c = int2binary[c_int]
    # d 存储对c的预测值
    d = np.zeros_like(c)
    # 设置尾插误差
    overallError = 0
    # 这个看不懂后面看
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        #这一批循环会相当于走完一个序列
        # 生成训练集和测试集，如：X [[1,0]] Y :[[1]]
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T
        # RNN的更新函数，将本轮的输入X与权重矩阵synapse_0相乘得到维度是1 * 16
        # 将上一轮的b_h-1和权重矩阵synapse_h相乘得到另一个1*16
        # 二者相加，得到新的输出（1*16）经过sigmod函数得到b_h(也就是下行中的layer_1)，b_h会传递到下一层，依次循环
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))
        # 得到的b_h与权重矩阵相乘然后经过sigmoid函数后，得到输出的结果
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))
        # 计算误差
        layer_2_error = y - layer_2
        # 下面是开始反向传播，记录了每一个时间步的误差与sigmod的乘积
        layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
        # 将输出值存储
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        # 将隐层单元存储，我们方便传递到下一层
        layer_1_values.append(copy.deepcopy(layer_1))
    # 这里是记录从最后一步反向传过来的误差，这也是RNN容易忘事和梯度消失的原因
    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        # 反向更新参数
        X = np.array([[a[position], b[position]]])
        # 倒着去除每一个隐层输出
        layer_1 = layer_1_values[-position - 1]
        # 取出相对的上一层的输出
        prev_layer_1 = layer_1_values[-position - 2]
        # 倒着取出每一层的误差
        layer_2_delta = layer_2_deltas[-position - 1]
        # 这里相当于算了大部分的w_hh的误差了
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(
            synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        # let's update all our weights so we can try again
        # synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
        # synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
        # synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1
        # 这里是计算对w_hk的导数，layer_2_delta已经计算过sigmoid的导数了，所以公式是对的
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        # 这是整个的w_hh  的误差，完全符合公式
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        # 这是对w_ih的导数计算，这里有点疑惑，计算w_ih为什么要把future步骤的误差算上？
        synapse_0_update += X.T.dot(layer_1_delta)
        # 更新来自之前的所有误差值
        future_layer_1_delta = layer_1_delta
    # 更新权重
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # print out progress
    if (j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print
        "------------"

