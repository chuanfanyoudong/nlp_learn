"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: Combinations.py
@time: 2019/2/28 0:07
"""


class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        if k == 0:
            return []
        if k == 1:
            return [[i] for i in range(1,n+1)]
        if k == n:
            return [[i for i in range(1, n+1)]]
        return self.combine(n-1, k) + [[n] + i for i in self.combine(n-1,k-1)]


if __name__ == '__main__':
    print(Solution().combine(4,2))