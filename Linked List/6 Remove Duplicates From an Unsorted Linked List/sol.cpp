/*
qn:https://www.codingninjas.com/codestudio/problems/remove-duplicates-from-unsorted-linked-list_1069331?topList=love-babbar-dsa-sheet-problems&utm_source=website&utm_medium=affiliate&utm_campaign=450dsatracker
*/
#include <bits/stdc++.h>
using namespace std;

int main()
{

    return 0;
}

class Node
{
public:
    int data;
    Node *next;
    Node(int data)
    {
        this->data = data;
        this->next = NULL;
    }
};
Node *removeDuplicates(Node *head)
{
    // Write your code here
    unordered_map<int, int> mp;
    Node *front = head, *back = head;
    // mannnnn just this one case caused my code to give wrong output
    // ne careful when we should put node and node->next !==NULL
    while (front != NULL)
    {
        mp[front->data]++;
        if (mp[front->data] > 1)
        {
            back->next = front->next;
            front = back->next;
        }
        else
        {
            back = front;
            front = front->next;
        }
    }
    return head;
}