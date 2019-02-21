"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: VaildShudu.py
@time: 2019/2/21 20:59
"""
import collections


class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """

    def isValidSudoku(self, board):
        result = []
        for i , row in enumerate(board):
            for j ,c in enumerate(row):
                if c != ".":
                    for x in ((c, i), (j, c), (i / 3, j / 3, c)):
                        result.append(x)
        """
        对每一个c都查看作为行，列，九宫格时有没有重复的
        （c，i）表示行
        （j, c）表示列
        (i / 3, j / 3, c)表示九宫格
        最后 result + [1]为了防止全空格
        """
        # print(result)
        return max(collections.Counter(result + [1]).values())


if __name__ == '__main__':
    board = [
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
    # list_t  = [1,2,3,4]
    # print([i for i in list_t if i != 3])
    print(Solution().isValidSudoku(board))