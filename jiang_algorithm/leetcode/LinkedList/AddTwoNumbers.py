"""
输入两个链表，模拟加法运算
"""


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        注意返回的也要是一个链表
        就是一次相加，做一个标记为tag如果大于10，tag就==1，那在加下一位的时候就加额外的1
        当其中一个链表为None后，另一个依次加上就可以，最终还要判断一下tag的值，若为1则后面在加1
        这道题自己一直做不好，讲道理不应该的，
        """
        if l1 == None:
            return l2
        if l2 == None:
            return l1
        tag = 0
        du = pre = None
        while l1 != None or l2 != None:
            if l1:
                d1 = l1.val
            else:
                d1 = 0
            if l2:
                d2 = l2.val
            else:
                d2 = 0
            result = d1 + d2 + tag
            if result > 9:
                tag = 1
                result = result%10
            else:
                tag = 0
            if du is None:
                du = ListNode(result)
                # 下面这句话代表着，pre和du指向了同一个节点，如果pre.next 赋值为7那么du.next 也为7
                # 如果说pre = ListNode(1),那么只是说明了pre指向的节点变了，而du指向的节点没有变，还是原来那个
                pre = du
            else:
                lt = ListNode(result)
                pre.next = lt
                pre = lt
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        if tag == 1:
            pre.next = ListNode(1)
        return du

if __name__ == '__main__':
    l1 = ListNode(0)
    l2 = ListNode(7)
    l2.next = ListNode(3)
    resu = Solution().addTwoNumbers(l1, l2)
    print(resu.val)
