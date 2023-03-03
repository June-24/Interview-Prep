/*
Here's how the code works:

1. The partition function takes a reference to a vector v,
 the indices left and right specifying the range of elements
  to partition, and a pivot element. It uses two pointers to 
  traverse the vector from left to right and from right to left, 
  swapping elements that are in the wrong partition. The function 
  returns the index of the first element in the right partition.

2. The quickselect function takes a reference to a vector v, 
the indices left and right specifying the range of elements 
to search, and the k-th smallest element to find. It chooses
 a pivot element randomly from the range, partitions the vector 
 around the pivot, and determines the index of the pivot element.
  If the pivot element is the k-th smallest, the function returns it.
   Otherwise, it recursively calls itself on the appropriate partition
    based on the index of the pivot element relative to the target k-th smallest element.

3. In the main function, we initialize the input vector v and the target k-th
 smallest element k. We call quickselect with the appropriate arguments and print the result.
*/

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;

// Partition the input vector into two parts: one with elements smaller than pivot and the other with elements greater than pivot.
int partition(vector<int>& v, int left, int right, int pivot) {
    while (left <= right) {
        while (v[left] < pivot) {
            left++;
        }
        while (v[right] > pivot) {
            right--;
        }
        if (left <= right) {
            swap(v[left], v[right]);
            left++;
            right--;
        }
    }
    return left;
}

// Find the k-th smallest element in the input vector.
int quickselect(vector<int>& v, int left, int right, int k) {
    // Choose a pivot element randomly.
    int pivot = v[left + rand() % (right - left + 1)];

    // Partition the vector around the pivot.
    int index = partition(v, left, right, pivot);

    // Recursively select the k-th smallest element in the appropriate partition.
    if (left + k - 1 == index - 1) {
        return pivot;
    } else if (left + k - 1 < index - 1) {
        return quickselect(v, left, index - 2, k);
    } else {
        return quickselect(v, index, right, k - (index - left));
    }
}

int main() {
    vector<int> v = {10, 4, 5, 8, 6, 11, 26};
    int k = 3;

    int kth_smallest = quickselect(v, 0, v.size() - 1, k);

    cout << "The " << k << "-th smallest element in the input array is " << kth_smallest << endl;

    return 0;
}
