"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: RemoveDuplicatesfromSortedList2.py
@time: 2019/3/3 14:58
"""


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummp = pre = ListNode(0)
        dummp.next = head
        while head and head.next:
            if head.val == head.next.val:
                ## 注意下面循环while的判断顺序，要先判断有没有nex然后在判断相等
                while head.val == head.next.val and head and head.next:
                    head = head.next
                head = head.next
                pre.next = head
            else:
                pre = pre.next
                head = head.next
        return dummp.next
