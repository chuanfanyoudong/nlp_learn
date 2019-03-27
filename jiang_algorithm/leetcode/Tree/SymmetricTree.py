"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: SymmetricTree.py
@time: 2019/3/21 18:53
"""
"""
# 给出一棵树，看一下是不是左右完全对称的树
# 开心这道题自己完全没看其他人的解法做出来的，虽然速度一般般
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3
But the following [1,2,2,null,3,null,3] is not:
    1
   / \
  2   2
   \   \
   3    3
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root == None:
            return True
        return self.ismirror(root.left, root.right)

    def ismirror(self, left, right):
        # 如果左右两棵树都为空，返回True
        if left == None and right == None:
            return True
        # 如果左右只有一个为空，那么说明不相等，所以一定是False
        if left == None or right == None:
            return False
        # 如果左右两个节点值不相等，那么一定返回False
        if left.val != right.val:
            return False
        # 返回left的左边和right的右边的结果与left的右边和right的左边的结果的并
        return self.ismirror(left.left, right.right) and self.ismirror(left.right, right.left)


