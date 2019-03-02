"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: SetMatrixZeroes.py
@time: 2019/2/27 22:19
"""
test = [
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]

class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        if matrix == 0:
            return []
        zero_id =[]
        m = len(matrix)
        n = len(matrix[0])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    zero_id.append((i,j))
        row_set = set()
        col_set = set()
        for single_id in zero_id:
            row_set.add(single_id[0])
            col_set.add(single_id[1])
        for i in row_set:
            matrix[i] = [0]*n
        for j in col_set:
            for i in range(m):
                matrix[i][j] = 0
        return matrix

if __name__ == '__main__':
    print(Solution().setZeroes(test))