import unittest


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre, pre.next = self, head
        while pre.next and pre.next.next:
            a = pre.next
            b = a.next
            pre.next = b
            b.next = a
            a.next = b.next
            pre = a  ## 不要忘记改变pre节点的位置
        return pre.next


if __name__ == '__main__':
    unittest.main()
