/*
qn: https://www.codingninjas.com/codestudio/problems/unique-sorted-list_2420283?topList=love-babbar-dsa-sheet-problems&utm_source=website&utm_medium=affiliate&utm_campaign=450dsatracker

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

Node *uniqueSortedList(Node *head)
{
    // Write your code here.//basic method not working myan
    // Node *curr = head;
    // while (curr != NULL)
    // {
    //     if (curr->next != NULL)
    //     {
    //         if (curr->data == curr->next->data)
    //         {
    //             Node *temp = curr->next;
    //             while (temp->data == curr->data)
    //                 temp = temp->next;
    //             curr->next=temp;
    //         }
    //     }
    //     else
    //     {
    //         curr = curr->next;
    //     }
    // }
    // return head;

    //new method
    //best way myan no need to consecutively delete all 
    //just delete one at a time it will remove all lmao
    if(head==NULL)return head;
    Node * curr=head;
    while(curr->next!=NULL){
        if(curr->data==curr->next->data){
            Node * temp=curr->next;
            curr->next=temp->next;
            delete temp;
        }
        else{
            curr=curr->next;
        }
    }
    return head;

}