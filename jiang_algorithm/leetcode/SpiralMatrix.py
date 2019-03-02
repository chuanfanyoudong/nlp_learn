"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: SpiralMatrix.py
@time: 2019/2/23 21:02
"""


class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        result = []
        while matrix:
            result += matrix.pop(0)
            if matrix and matrix[0]:
                for i in matrix:
                    result.append(i.pop())
            if matrix:
                result += (matrix.pop()[::-1])
            if matrix and matrix[-1]:
                for i in matrix[::-1]:
                    result.append(i.pop(0))

        return result

if __name__ == '__main__':
    test_list = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
    print(Solution().spiralOrder(test_list))