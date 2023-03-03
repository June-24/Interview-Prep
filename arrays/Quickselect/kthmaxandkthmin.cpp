/*
link to this problem:https://www.codingninjas.com/codestudio/problems/kth-smallest-and-largest-element-of-array_1115488?topList=love-babbar-dsa-sheet-problems&utm_source=website&utm_medium=affiliate&utm_campaign=450dsatracker
In this implementation,
 we use the partition
  function to find both
   the kth smallest and kth
   largest elements. We pass in the index of the kth
   smallest element as an additional parameter to the partition
   function, and return the element at that index if it is the pivot
   element. If the pivot index is less than k, we know that the kth
   smallest element must be in the right half of the array, so we call
   the partition function on the right half of the array. If the pivot index
   is greater than k, we know that the kth smallest element must be in the left
   half of the array, so we call the partition function on the left half of the array.
    We do the same thing to find the kth largest element, using the index kthLargestIndex
    instead of kthSmallestIndex.
*/

#include <iostream>
#include <vector>

using namespace std;

int partition(vector<int> &nums, int left, int right, int &k)
{
    int pivot = nums[left];
    int i = left + 1, j = right;
    // basic swap
    while (i <= j)
        if (nums[i] < pivot)
            i++;
        else if (nums[j] >= pivot)
            j--;
        else
            swap(nums[i++], nums[j--]);
    swap(nums[left], nums[j]);
    // if the pivot is the position where the min or max is supposed to be then return that
    if (j == k)
        return nums[j];
    // suppose k is big then then we take the right side and search there ie from j+1 to right
    else if (j < k)
        return partition(nums, j + 1, right, k);
    // if k is less than j then we need to search left side from left to j-1
    else
        return partition(nums, left, j - 1, k);
}

void findKthSmallestAndLargest(vector<int> &nums, int k, int &kthSmallest, int &kthLargest)
{
    int n = nums.size();
    int kthSmallestIndex = k - 1; // yes k-1 you got that right index starts feom 0 so we need to reduce 1
    // critical point for finding the kth max we are converting it to min terms and sending that thats very important
    int kthLargestIndex = n - k;
    kthSmallest = partition(nums, 0, n - 1, kthSmallestIndex);
    kthLargest = partition(nums, 0, n - 1, kthLargestIndex);
}

int main()
{
    vector<int> nums = {9, 1, 5, 2, 8, 3};
    int k = 3;
    int kthSmallest, kthLargest;
    findKthSmallestAndLargest(nums, k, kthSmallest, kthLargest);
    cout << "The " << k << "th smallest element is " << kthSmallest << endl;
    cout << "The " << k << "th largest element is " << kthLargest << endl;
    return 0;
}
