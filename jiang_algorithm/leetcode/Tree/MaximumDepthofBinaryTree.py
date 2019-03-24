#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com/chuanfanyoudong
@software: PyCharm
@file: MaximumDepthofBinaryTree.py
@time: 2019/3/24 14:58
"""

"""
# 返回一棵树的最大深度
Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Note: A leaf is a node with no children.

Example:

Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its depth = 3.
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        # 考虑用递归，递归公式为树的深度为左树深度和右树深度的最大值加1
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        return max(self.maxDepth(root.left) , self.maxDepth(root.right)) + 1
        # self.helper(root, max_deepth)

    def maxDepth2(self, root):
        # 广度优先搜索，非递归
        if root == None:
            return 0
        q = [root]
        max_deepth = 0
        while q != []:
            max_deepth += 1
            for i in range(len(q)):
                # 因为range的范围是此时q的长度，此时q的长度就等于这一层的节点个数
                if q[0].left:
                    # 因为后面有del函数，所以每次都是q[0],而不是q[i]
                    q.append(q[0].left.val)
                if q[0].right:
                    q.append(q[0].right)
                del q[0]
        return max_deepth