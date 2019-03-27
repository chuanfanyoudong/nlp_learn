"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: UniqueBinarySearchTrees.py
@time: 2019/3/18 22:55
"""

# 给一个数字输出它所有可能的二叉搜索树
# 二叉搜索树就是左边的都比root小，右边的都比root大

"""
Input: 3
Output: 5
Explanation:
Given n = 3, there are a total of 5 unique BST's:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
"""

"""
思路：求数量的题目，直接抗动态规划，难点在于总结规律
"""

class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 初始化n=0,1,2的值
        res = [1,1,2]
        # 如果n < 3则不用进行迭代，直接输出
        if n < 3:
            return res[n]
        # 如果n >= 3，则根据公式
        # dp[n]=dp[0]*dp[n-1]+dp[1]*dp[n-2]+......+dp[n-1]*dp[0]
        else:
            # 下面是一个技巧，先把res所有的都赋值为0，除了前三个。
            # 这样是方便后面运算
            res += [0 for i in range(n-2)]
            # 两个循环搞定上面提到的公式
            for i in range(3,n + 1):
                # 每一个位置的结果
                for j in range(i):
                    #都是一次循环相加求和式
                    res[i] += res[j]*res[i - j - 1]
        return res[n]