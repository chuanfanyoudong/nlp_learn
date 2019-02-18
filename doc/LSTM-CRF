## LSTM-CRF

## 链接

- https://zhuanlan.zhihu.com/p/27338210
- https://blog.csdn.net/cuihuijun1hao/article/details/79405740

## trick

- LSTM没法对状态之间的关系做约束，会出现BB这种情况，所以后面加上CRF会约束这种情况出现，但是理论上还是可能出现的

- 这里用一种简便的方法，对于到词w_{i+1}的路径，可以先把到词w_i的logsumexp计算出来

- 设隐藏状态的数量为K

- LSTM的输出为维度为1*K，代表着发射概率

- 同时会自己定义一个转移概率，维度是K*k

- 定义一个得分函数，就是如果某一条可能的标记结果得分为
    
        S(X,y) = 经过这条结果的所有转移概率和发射概率之和

- 优化目标是让S(X,y) 在所有可能的得分函数中占的比重尽可能的高，会用到log方法


    
