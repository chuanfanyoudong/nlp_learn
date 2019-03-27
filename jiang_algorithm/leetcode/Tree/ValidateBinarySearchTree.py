"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: ValidateBinarySearchTree.py
@time: 2019/3/20 22:00
"""



"""

给你一颗二叉树，你来判断其是否属于二叉搜索树

Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
Example 1:

Input:
    2
   / \
  1   3
Output: true
Example 2:

    5
   / \
  1   4
     / \
    3   6
Output: false
Explanation: The input is: [5,1,4,null,null,3,6]. The root node's value
             is 5 but its right child's value is 4.
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
    # 最好理解的一个方法是先对这棵树进行中序遍历，然后，依次比较遍历后的结果，后一个值比前一个值大就可以
    # def isVailable(self, root):
        if root is None:
            return True
        if root.left == None and root.right == None:
            return True
        self.result = []
        self.helper(root)
        for i in range(1,len(self.result)):
            if self.result[i] <= self.result[i-1]:
                return False
        return True

    def helper(self, root):
        if root == None:
            return None
        self.helper(root.left)
        self.result.append(root.val)
        self.helper(root.right)

# 人类的智慧啊，上一种方法需要额外的空间
# 因此人们想到了一种不需要额外空间的方法
class Solution(object):
    # def
    pre = None

    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        # 最好理解的一个方法是先对这棵树进行中序遍历，然后，依次比较遍历后的结果，后一个值比前一个值大就可以
        # 这里就是设计一个pre值，记录pre值，保证pre老是小于当前root
        if root is None:
            return True
        # 输出左子树的遍历结果
        Bool = self.isValidBST(root.left)
        if self.pre != None:
            Bool = Bool and (self.pre.val < root.val)
        # 将当前节点赋值给pre
        self.pre = root
        # 遍历右边节点的时候，相当于要root小于右边的所有值
        Bool = Bool and self.isValidBST(root.right)
        return Bool