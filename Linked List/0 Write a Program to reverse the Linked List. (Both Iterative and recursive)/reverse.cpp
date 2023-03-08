/*
qn: https://www.codingninjas.com/codestudio/problems/reverse-the-singly-linked-list_799897?topList=love-babbar-dsa-sheet-problems&utm_source=website&utm_medium=affiliate&utm_campaign=450dsatracker
aim to do in n time and 1 space complexity

1->2->3->4->5->6->7
prev=null
curr=head

curr=curr->next
head->next=prev
prev=head
head=curr

*/

// LinkedListNode<int> *reverseLinkedList(LinkedListNode<int> *head)
// {
//     // Write your code here
//     LinkedListNode<int> *prev = NULL;
//     LinkedListNode<int> *curr = head;

//     while (head != NULL)
//     {
//         curr = curr->next;
//         head->next = prev;
//         prev = head;
//         head = curr;
//     }
//     return prev;
// }