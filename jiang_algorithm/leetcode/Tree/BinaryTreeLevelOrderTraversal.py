#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com/chuanfanyoudong
@software: PyCharm
@file: Binary Tree Level Order Traversal.py
@time: 2019/3/24 14:28
"""

"""
# 给出一个二叉树，返回其层次遍历的结果
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]

"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    """
    看到这个题首先的一个思路就是用栈，用一个栈代表节点，用另一个栈代表层数，
    然后发现做不出来，看到大佬们的思路是用先序遍历
    """
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        res = []
        return self.helper(root, 0, res)

    def helper(self, root, level, res):
        # 做一个先序遍历
        # 先做一个判断是不是新的一层，如果是新的一层需要在res中加一个[]
        if root:
            if len(res) < level + 1:
                res.append([])
            # 这一层的一个节点root的值被添加进去
            res[level].append(root.val)
            self.helper(root.left, level + 1, res)
            self.helper(root.right, level + 1, res)
        return res
