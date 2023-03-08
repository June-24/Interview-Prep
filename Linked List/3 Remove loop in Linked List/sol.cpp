/*
qn: https://practice.geeksforgeeks.org/problems/remove-loop-in-linked-list/1

*/
//{ Driver Code Starts
// driver code

#include <bits/stdc++.h>
using namespace std;

struct Node
{
    int data;
    Node *next;

    Node(int val)
    {
        data = val;
        next = NULL;
    }
};

void loopHere(Node *head, Node *tail, int position)
{
    if (position == 0)
        return;

    Node *walk = head;
    for (int i = 1; i < position; i++)
        walk = walk->next;
    tail->next = walk;
}

bool isLoop(Node *head)
{
    if (!head)
        return false;

    Node *fast = head->next;
    Node *slow = head;

    while (fast != slow)
    {
        if (!fast || !fast->next)
            return false;
        fast = fast->next->next;
        slow = slow->next;
    }

    return true;
}

int length(Node *head)
{
    int ret = 0;
    while (head)
    {
        ret++;
        head = head->next;
    }
    return ret;
}
// method one
//  class Solution
//  {
//  public:
//      Node *get_pivot(Node *head)
//      {
//          unordered_map<Node *, int> mp;
//          Node *curr;
//          while (curr != NULL)
//          {
//              mp[curr]++;
//              if (mp[curr] == 2)
//                  return curr;
//              curr = curr->next;
//          }
//          return NULL;
//      }
//      void removeLoop(Node *head)
//      {
//          // code here
//          // just remove the loop without losing any nodes
//          Node *curr = get_pivot(head);
//          if (curr == NULL)
//              return;
//          Node *tra = head;
//          int ct = 0;
//          while (ct != 2)
//          {
//              if (tra->next == curr)
//              {
//                  ct++;
//                  if (ct == 2)
//                      break;
//              }
//              tra = tra->next;
//          }
//          tra->next=NULL;
//          return;
//      }
//  };
// new way
class Solution
{
public:
    // Function to remove a loop in the linked list.
    void removeLoop(Node *head)
    {
        // code here
        // just remove the loop without losing any nodes
        if(head==NULL)return;
        unordered_map<Node *, int> mp;
        mp[head]++;
        Node * curr=head;
        while(curr->next!=NULL){
            mp[curr->next]++;
            if(mp[curr->next]==2){
                curr->next=NULL;
                return;
            }
            curr=curr->next;
        }
    }
};
int main()
{
    int t;
    cin >> t;
    while (t--)
    {
        int n, num;
        cin >> n;

        Node *head, *tail;
        cin >> num;
        head = tail = new Node(num);

        for (int i = 0; i < n - 1; i++)
        {
            cin >> num;
            tail->next = new Node(num);
            tail = tail->next;
        }

        int pos;
        cin >> pos;
        loopHere(head, tail, pos);

        Solution ob;
        ob.removeLoop(head);

        if (isLoop(head) || length(head) != n)
            cout << "0\n";
        else
            cout << "1\n";
    }
    return 0;
}
