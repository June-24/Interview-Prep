/*
qn: https://www.codingninjas.com/codestudio/problems/deleting-and-adding-the-last-node_1170051?topList=love-babbar-dsa-sheet-problems&utm_source=website&utm_medium=affiliate&utm_campaign=450dsatracker

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

Node *delAddLastNode(Node *head)
{
    // Write your code here.
    // bugs in this code....................
    // if (head->next == NULL || head == NULL)
    //     return head;
    // Node *temp1 = head->next, *temp2 = head;
    // while (temp1->next != NULL)
    // {
    //     temp1 = temp1->next;
    //     temp2 = temp2->next;
    // }
    // temp2->next = NULL;
    // temp1->next = head;
    // return temp1;

    // minor updations..............
    if (head == NULL || head->next == NULL)
        return head;
    Node *temp1 = head, *temp2 = NULL;
    while (temp1->next != NULL)
    {
        temp2 = temp1;
        temp1 = temp1->next;
    }
    temp2->next = NULL;
    temp1->next = head;
    return temp1;
}
