"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: UniquePaths.py
@time: 2019/2/25 21:41
"""


class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """

        result = [[1 for i in range(n)] for j in range(m)]
        for i in range(1,m):
            for j in range(1,n):
                result[i][j] = result[i-1][j] + result[i][ j-1]
        return result[-1][-1]

        # if m == 1 or n == 1:
        #     return 1
        # return self.uniquePaths(m-1, n) + self.uniquePaths(m, n-1)
if __name__ == '__main__':
    print(Solution().uniquePaths(4,4))