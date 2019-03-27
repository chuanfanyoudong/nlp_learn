"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: BinaryTreeInorderTraversal.py
@time: 2019/3/11 23:53
"""
"""
中序遍历二叉树
"""


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# https://www.cnblogs.com/icekx/p/9127569.html

class Solution(object):
    # 下面是递归方法
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        self.recursive_inorder(root, res)
        return res

    def recursive_inorder(self, root, res):
        # 只要当前节点是不是空，他就等于左子节点的结果 + 当前节点 + 右子节点的结果
        if root:
            self.recursive_inorder(root.left, res)
            res.append(root.val)
            self.recursive_inorder(root.right, res)


    # 下面是非递归方法
    def inorderTraversal2(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        stack = []
        res = []
        pos = root
        while pos is not None or len(stack) > 0:
            if pos is not None:# 这个if内容的作用是如果pos的左子节点不为空，那么就压入栈，
                stack.append(pos)  # 压入栈
                pos = pos.left  # 把当前节点的左子节点赋值给当前节点
            else:# 这里的作用是发现这个节点左子节点是空了，就把当前节点从栈里面弄出来
                pos = stack.pop()  # 把当前节点作为根节点从栈里面弄出来
                res.append(pos.val)  # 打印这个节点
                pos = pos.right  # 然后看看它的右节点，如果也是空，就把上一层的根节点弄出来，如果不是空，就走上面的
                #  逻辑一直走到最最左下角
        return res