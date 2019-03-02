"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: UniquePaths2.py
@time: 2019/2/25 22:03
"""


class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        if obstacleGrid == []:
            return 0
        if obstacleGrid == [[1]]:
            return 0
        if obstacleGrid[-1][-1] == 1 or obstacleGrid[0][0] == 1:
            return 0
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        result = [[0 for i in range(n)] for j in range(m)]
        for i in range(m-1,-1,-1):
            if obstacleGrid[i][-1] == 0:
                result[m - i - 1][0] = 1
            if obstacleGrid[i][-1] == 1:
                break
        for j in range(n-1,-1,-1):
            if obstacleGrid[-1][j] == 0:
                result[0][n - j -1] = 1
            if obstacleGrid[-1][j] == 1:
                break
        print(result)
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[m - i - 1][n - j - 1] == 1:
                    result[i][j] = 0
                else:
                    if obstacleGrid[m - i][n - j -1] == 1 and obstacleGrid[m - i - 1][n - j] == 1:
                        result[i][j] = 0
                    elif obstacleGrid[m - i][n - j - 1] == 1:
                        result[i][j] = result[i][j - 1]
                    elif obstacleGrid[m-i-1][n-j] == 1:
                        result[i][j] = result[i - 1][j]
                    else:
                        result[i][j] = result[i - 1][j] + result[i][j - 1]
        return result[-1][-1]


if __name__ == '__main__':
    test = [[0,0,0,0,0],[0,0,0,0,1],[0,0,0,1,0],[0,0,0,0,0]]
    print(Solution().uniquePathsWithObstacles(test))