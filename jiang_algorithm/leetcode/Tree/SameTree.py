"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: SameTree.py
@time: 2019/3/21 0:24
"""
"""
# 判断两棵树是否相等，很容易想到用递归
Given two binary trees, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical and the nodes have the same value.

Example 1:

Input:     1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

Output: true
Example 2:

Input:     1         1
          /           \
         2             2

        [1,2],     [1,null,2]

Output: false
Example 3:

Input:     1         1
          / \       / \
         2   1     1   2

        [1,2,1],   [1,1,2]

Output: false

"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        # 如果两个节点都为None发挥True
        if p == None and q == None:
            return True
        # 如果两个节点的值相等，且都不为None
        if q and p and q.val == p.val:
            # 那么返回他们的左节点是否相等和有右节点是否相等的与运算
            # 其实就是当前节点相等的情况下，判断左右子树是否相等
            return self.isSameTree(q.left, p.left) and self.isSameTree(q.right, p.right)
        # 其他情况返回False
        return False
