/*
qn: https://www.codingninjas.com/codestudio/problems/detect-the-first-node-of-the-loop_1112628?topList=love-babbar-dsa-sheet-problems&utm_source=website&utm_medium=affiliate&utm_campaign=450dsatracker

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

Node *firstNode(Node *head)
{
    //    Write your code here.
    unordered_map<Node*, int> mp;
    Node* curr=head;
    while(curr!=NULL){
        mp[curr]++;
        if(mp[curr]==2)return curr;
    curr=curr->next;
    }
    return NULL;

}