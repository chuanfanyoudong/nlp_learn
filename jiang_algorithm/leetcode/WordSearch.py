"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: WordSearch.py
@time: 2019/3/2 19:49
"""

test_list = [["C","A","A"],["A","A","A"],["B","C","D"]]

word = "AAB"

class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        if board == []:
            return False
        word = list(word)

        m = len(board)
        n = len(board[0])
        for i in range(m):
            for j in range(n):
                tmp_board = list(board)
                if self.dfs(tmp_board, i, j, word):
                    return True
        else:
            return False

    def dfs(self, board, i,j, word):
        ## 不用tag_list,直接修改board就可以
        if len(word) == 0:
            return True
        m = len(board)
        n = len(board[0])
        if i < 0 or i > m-1 or j <0 or j > n-1 or board[i][j] != word[0]:
            return False
        # if board[i][j] == word[0]:
        tmp = board[i][j]
        board[i][j] = "#"
        res = self.dfs(board,  i+1,j,word[1:]) or self.dfs(board, i-1,j ,word[1:]) or self.dfs(board,  i, j+1, word[1:])or self.dfs(board, i,j-1, word[1:])
        board[i][j] = tmp
        return res

if __name__ == '__main__':


    a = [1,2,3]
    b = [i for i in a]
    a[0] = 2
    print(a)
    print(b)
    print(Solution().exist(test_list, word))