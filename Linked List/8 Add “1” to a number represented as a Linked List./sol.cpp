/*
qn: https://practice.geeksforgeeks.org/problems/add-1-to-a-number-represented-as-linked-list/1
*/

//{ Driver Code Starts
// Initial template for C++

#include <bits/stdc++.h>
using namespace std;

struct Node
{
    int data;
    struct Node *next;

    Node(int x)
    {
        data = x;
        next = NULL;
    }
};

void printList(Node *node)
{
    while (node != NULL)
    {
        cout << node->data;
        node = node->next;
    }
    cout << "\n";
}

class Solution
{
public:
    Node *reverse(Node *head)
    {
        Node *prev = NULL;
        Node *curr = head;
        while (curr != NULL)
        {
            curr = curr->next;
            head->next = prev;
            prev = head;
            head = curr;
        }
        return prev;
    }
    Node *addOne(Node *head)
    {
        // Your Code here
        // return head of list after adding one
        /*
        doing in basic way is simple just make that number
         and add one to it and then put it back

         need to see a better way
        */
        // # NOTE --> Do not try solving by converting linked list
        //  into an
        // integer cause linked may be even longer than long long
        head = reverse(head);
        if (head->data < 9)
        {
            head->data++;
            head = reverse(head);
            return head;
        }
        else
        {
            int carry = 1;
            Node *temp = head;
            Node *prev = NULL;
            while (temp != NULL && carry != 0)
            {
                int sum = temp->data + carry;
                temp->data = sum % 10;
                carry = sum / 10;
                prev = temp;
                temp = temp->next;
            }
            if (carry > 0)
            {
                prev->next = new Node(1);
            }
            head = reverse(head);
            return head;
        }
    }
};

int main()
{
    int t;
    cin >> t;
    while (t--)
    {
        string s;
        cin >> s;

        Node *head = new Node(s[0] - '0');
        Node *tail = head;
        for (int i = 1; i < s.size(); i++)
        {
            tail->next = new Node(s[i] - '0');
            tail = tail->next;
        }
        Solution ob;
        head = ob.addOne(head);
        printList(head);
    }
    return 0;
}

// } Driver Code Ends