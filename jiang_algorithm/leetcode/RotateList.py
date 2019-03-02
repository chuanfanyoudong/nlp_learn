"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: RotateList.py
@time: 2019/2/25 20:59
"""

print(1%4)
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # @param head, a ListNode
    # @param k, an integer
    # @return a ListNode
    def rotateRight(self, head, k):
        if k == 0:
            return head
        if head == None:
            return head
        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        count = 0
        while p.next:
            p = p.next
            count += 1
        p.next = dummy.next
        step = count - (k % count)
        for i in range(0, step):
            p = p.next
        head = p.next
        p.next = None
        return head
