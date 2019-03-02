"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: RotateImage.py
@time: 2019/2/22 21:22
"""


class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        have_dowm = []
        num_row = len(matrix)
        for i, row in enumerate(matrix):
            for j, token in enumerate(row):
                if i < j:
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
                    # tmp = matrix[i][j]
                    # matrix[i][j] = matrix[j][num_row - 1 - i]
                    # matrix[j][num_row -1 - i] = tmp
                    # have_dowm.append((i,j))
                    # have_dowm.append((j, num_row -1 - i))
        for i in matrix:
            i.reverse()


if __name__ == '__main__':
    print(sorted(["a","c","b"]))
    print(Solution().rotate([
  [1,2,3],
  [4,5,6],
  [7,8,9]
]))