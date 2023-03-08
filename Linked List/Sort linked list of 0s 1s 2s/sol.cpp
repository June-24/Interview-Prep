/*
qn: https://www.codingninjas.com/codestudio/problems/sort-linked-list-of-0s-1s-2s_1071937?topList=love-babbar-dsa-sheet-problems&utm_source=website&utm_medium=affiliate&utm_campaign=450dsatracker

*/

/********************************
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

********************************/
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
Node *sortList(Node *head)
{
    // Write your code here.
    //list contains 0s 1s and 2s 
    //use dutch national flag algo
    //how tf will you use this man its a LL not an array 
    // Node * start=head,mid=head,end=
    //no point doing the dutch flag you see that tho
    int zero=0,one=1,two=0;
    Node * curr=head;
    while(curr!=NULL){
        if(curr->data==0)zero++;
        if(curr->data==1)one++;
        if(curr->data==2)two++;
        curr=curr->next;

    }
    curr=head;
    while(zero--){
        curr->data=0;
        curr=curr->next;
    }
    while(one--){
        curr->data=1;
        curr=curr->next;
    }
    while(two--){
        curr->data=2;
        curr=curr->next;
    }
}
