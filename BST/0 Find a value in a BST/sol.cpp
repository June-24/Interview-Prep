/*
qn: https://www.codingninjas.com/codestudio/problems/find-a-value-in-bst_1170063?topList=love-babbar-dsa-sheet-problems&utm_source=website&utm_medium=affiliate&utm_campaign=450dsatracker&leftPanelTab=1
*/

#include <bits/stdc++.h> 
/************************************************************

    Following is the TreeNode class structure

    template <typename T>
    class TreeNode {
       public:
        T data;
        TreeNode<T> *left;
        TreeNode<T> *right;

        TreeNode(T data) {
            this->data = data;
            left = NULL;
            right = NULL;
        }
    };

************************************************************/
int flag=0;
bool tra(TreeNode <int> * root, int key){
    if(root){
        if(root->data>key){
            tra(root->left,key);
        } else if (root->data < key) {
          tra(root->right, key);
        } else if (root->data == key){
          return true;
        }
    }
    else{
        return false;
    }
}

bool findNode(TreeNode <int> * root, int key) {
    // Write your code here.
    return tra(root,key);
}