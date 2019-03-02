"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: Searcha2DMatrix.py
@time: 2019/2/27 22:31
"""
test_list = matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """

        if matrix == 0:
            return 0
        m = len(matrix)
        n = len(matrix[0])
        row = m-1
        clo = 0
        while row >= 0 and clo < n:
            if matrix[row][clo] == target:
                return 1
            elif matrix[row][clo] > target:
                row -= 1
            else:
                clo += 1
        return 0

if __name__ == '__main__':
    print(Solution().searchMatrix(test_list, 5))