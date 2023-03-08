/*
qn: https://www.codingninjas.com/codestudio/problems/detect-and-remove-cycle_920523?topList=love-babbar-dsa-sheet-problems&utm_source=website&utm_medium=affiliate&utm_campaign=450dsatracker

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
bool detectAndRemoveCycle(Node *head)
{
    // Write your code here
    unordered_map<Node *, int> mp;
    mp[head]++;
    Node *curr = head;
    while (curr->next != NULL)
    {
        mp[curr->next]++;
        if (mp[curr->next] == 2)
        {
            curr->next = NULL;
            return true;
        }
        curr = curr->next;
    }
    return false;
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
