"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: UniqueBinarySearchTreesII.py
@time: 2019/3/20 21:40
"""

"""

要求输入一个数字n，把1到n能组成二叉搜索树全部表示出来，每棵树用广度优先遍历好了
遇到求数量的动态规划好了啊
但是这个让把所有的都显示出来，果断递归

Input: 3
Output:
[
  [1,null,3,2],
  [3,2,null,1],
  [3,1,null,null,2],
  [2,1,3],
  [1,null,2,null,3]
]
Explanation:
The above output corresponds to the 5 unique BST's shown below:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
"""


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def generateTrees(self, n):
        if n == 0:
            return []
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        return self.helper(1, n)

    def helper(self, start, end):
        result = []
        # 先判断边界，如果n为0，则直接result里面包括一个null就好
        if end < start:
            result.append(None)
            return result
        # 这个题中相当于先固定root，然后左子树的方法数和右子树的方法数笛卡尔积
        # 这里的i就是固定的根节点，因为题目中提到了没有0，所以从1开始

        for i in range(start,end+1):
            # 递归方法求出左子树
            lefttree = self.helper(start, i-1)
            #递归方法求出右子树
            righttree = self.helper(i + 1, end)
            # 下面就是左子树的每一个方法与右子树的每一个方法求笛卡尔积
            for single_left in lefttree:
                for single_right in righttree:
                    root = TreeNode(i)
                    root.left = single_left
                    root.right = single_right
                    result.append(root)
        return result
