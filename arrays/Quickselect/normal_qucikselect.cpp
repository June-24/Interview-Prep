/*

worst case is n^2 so be careful
used for finding the kth minimun element


Quickselect is a very useful algorithm with many applications. Here are a few examples:

Finding the kth smallest or largest element in an array: This is the main application of
 Quickselect, as we've already discussed.

Finding the median of an array: The median is simply the kth smallest element, where k is
 (n + 1) / 2 if n is odd and n / 2 if n is even. Therefore, Quickselect can be used to find 
 the median of an array in linear time.

Finding the top k elements in an array: By using Quickselect to find the kth smallest element,
 we can easily find the top k elements in an array in linear time.

Approximate nearest neighbor search: In high-dimensional data sets, exact nearest neighbor
 search can be computationally expensive. However, approximate nearest neighbor search algorithms can use Quickselect as a subroutine to efficiently search for nearby data points.

Quicksort: Quicksort is a popular sorting algorithm that uses the same partition function
 as Quickselect. Therefore, understanding Quickselect is essential for understanding Quicksort.

Overall, Quickselect is a versatile algorithm that has many practical applications in computer science and data analysis.
*/



#include <iostream>
#include <vector>

using namespace std;

int partition(vector<int> &arr, int l, int r)
{
    int pivot = arr[l];
    int i = l + 1;
    int j = r;
    while (i <= j)
    {
        while (i <= j && arr[i] < pivot)
            i++;
        while (i <= j && arr[j] > pivot)
            j--;
        if (i <= j)
        {
            swap(arr[i], arr[j]);
            i++;
            j--;
        }
    }
    swap(arr[l], arr[j]);
    return j;
}

int quickselect(vector<int> &arr, int l, int r, int k)
{
    if (l == r)
        return arr[l];
    int pos = partition(arr, l, r);
    if (pos == k - 1)
        return arr[pos];
    else if (pos > k - 1)
        return quickselect(arr, l, pos - 1, k);
    else
        return quickselect(arr, pos + 1, r, k);
}

int main()
{
    vector<int> arr = {9, 1, 5, 2, 8, 3};
    int k = 3;
    int kth_smallest = quickselect(arr, 0, arr.size() - 1, k);
    cout << kth_smallest << endl; // Output: 5
    return 0;
}
