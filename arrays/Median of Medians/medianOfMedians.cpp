/*
In this implementation, the MedianOfMedians function takes a vector of integers v
 and the index k of the kth smallest element to find as inputs. The function recursively 
 partitions the array into groups of five elements each, computes the medians of each group,
  and then selects the median of medians as the pivot element. Finally, the function partitions
   the array around the pivot and recursively selects one of the resulting subarrays until the
    kth smallest element is found.

The main function in this implementation creates an example array and finds the 3rd
 smallest element using the MedianOfMedians function.

*/
/*
dry run
We want to find the 3rd smallest element in the array v, which is 6.

1.The function is called with v = {10, 4, 5, 8, 6, 11, 26} and k = 3.
2.The size of the input array is 7, which is greater than 1, so we proceed with the algorithm.
3.We partition the input array into groups of five elements each:
4.For each group, we sort the elements and select the median:
5.We recursively call MedianOfMedians with medians as the input array and the index of the median of medians (which is medians.size() / 2) as the target kth smallest element:
6.The MedianOfMedians function is called again with the input array {5, 26} and k = 0 (since the median of medians has index 0 in this array).
7.The size of the input array is 2, which is greater than 1, so we proceed with the algorithm.
8.We partition the input array into groups of five elements each:
9.For each group, we sort the elements and select the median:
10.The median of medians is 5, so we set pivot to 5 and return it to the previous call of MedianOfMedians.
11.We partition the original input array {10, 4, 5, 8, 6, 11, 26} around the pivot 5:
12.We recursively call MedianOfMedians with left as the input array and k = 2 (since we've eliminated one element from consideration):
13.The size of the input array is 1, so we return the only element in the array (4) to the previous call of MedianOfMedians.
14.Since k is less than the number of elements in left (which is 1), we return the value of pivot (which is 5) to the original call of MedianOfMedians.
15.The 3rd smallest element in the input array {10, 4, 5, 8, 6, 11, 26} is 6, which is returned to the main function.
*/


#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int MedianOfMedians(vector<int> &v, int k)
{
    int n = v.size();
    if (n == 1)
    {
        return v[0];
    }
    vector<int> medians;
    for (int i = 0; i < n; i += 5)
    {
        vector<int> temp;
        for (int j = i; j < i + 5; j++)
        {
            if (j < n)
            {
                temp.push_back(v[j]);
            }
        }
        sort(temp.begin(), temp.end());
        int m = temp[temp.size() / 2];
        medians.push_back(m);
    }
    int pivot = MedianOfMedians(medians, medians.size() / 2);
    vector<int> left, right;
    for (int i = 0; i < n; i++)
    {
        if (v[i] < pivot)
        {
            left.push_back(v[i]);
        }
        else if (v[i] > pivot)
        {
            right.push_back(v[i]);
        }
    }
    int num_left = left.size();
    if (k < num_left)
    {
        return MedianOfMedians(left, k);
    }
    else if (k == num_left)
    {
        return pivot;
    }
    else
    {
        return MedianOfMedians(right, k - num_left - 1);
    }
}

int main()
{
    vector<int> v{10, 4, 5, 8, 6, 11, 26};
    int k = 4;
    int kth_smallest = MedianOfMedians(v, k - 1);
    cout << "The " << k << "th smallest element is " << kth_smallest << endl;
    return 0;
}