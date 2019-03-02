"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: MinimumPathSum.py
@time: 2019/2/26 21:26
"""
test = [
  [1,3,1],
  [1,5,1],
  [4,2,1]
]


class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if grid == []:
            return None
        m = len(grid)
        n = len(grid[0])
        i = 0
        j = 0
        for i in range(1,m):
            grid[i][0] = grid[i][0] + grid[i - 1][0]
        for j in range(1,n):
            grid[0][j] += grid[0][j-1]
        for i in range(1,m):
            for j in range(1,n):
                grid[i][j] = min(grid[i-1][j],grid[i][j-1]) + grid[i][j]
        return grid[m-1][n-1]