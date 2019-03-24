#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com/chuanfanyoudong
@software: PyCharm
@file: ConstructBinaryTreefromPreorderandInorderTraversal.py
@time: 2019/3/24 15:45
"""

"""
# 给出一棵树的先序遍历和中序遍历，返还整棵树
Given preorder and inorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

For example, given

preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
Return the following binary tree:

    3
   / \
  9  20
    /  \
   15   7

"""


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    # 递归方法做
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        # 如果长度为空，直接返回None
        if len(preorder) == 0:
            return None
        # 根据先序遍历第一个值就是root确定root
        root = TreeNode(preorder[0])
        # 根据中序遍历root的位置确定左右两颗树的长度
        index = inorder.index(preorder[0])
        # 遍历左树，注意index的具体值
        root.left = self.buildTree(preorder[1:index + 1], inorder[:index])
        # 遍历右树
        root.right = self.buildTree(preorder[index + 1:], inorder[index + 1:])
        return root
