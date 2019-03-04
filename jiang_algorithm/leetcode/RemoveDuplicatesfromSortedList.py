"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: RemoveDuplicatesfromSortedList.py
@time: 2019/3/3 15:13
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
        dummpy = pre = ListNode(0)
        pre.next = head
        while head and head.next:
            if head.val == head.next.val:
                while head and head.next and head.val == head.next.val:
                    head = head.next
                pre.next = head
            else:
                pre.next = head
                head = head.next
        return dummpy.next

