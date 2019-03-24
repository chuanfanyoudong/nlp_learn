#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com/chuanfanyoudong
@software: PyCharm
@file: BinaryTreeZigzagLevelOrderTraversal.py
@time: 2019/3/24 14:51
"""

'''
# 返回一棵树的层次遍历，但是要求Z字形遍历，就是如果上一层是从左到右，那么下一层就是从右到左
Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]

'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    # 根据普通层次遍历的思路，考虑遍历完后，处理一下
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        res = []
        self.helper(root, 0, res)
        for i in range(len(res)):
            if i%2 != 0:
                res[i] = res[i][::-1]
        return res

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




