/*
qn: https://practice.geeksforgeeks.org/problems/kadanes-algorithm-1587115620/1
see the video by anuj bhaiya, its the best one yetm the one by love babbar is not really good
the feel of it is very important
*/
#include <bits/stdc++.h>
using namespace std;
int kadanes(vector<int> &v)
{
    // max_so_far is the max final ans
    int max_so_far = INT_MIN;
    int max_ending_here = 0; // important point
    for (int i = 0; i < v.size(); i++)
    {
        // add the latest element to the new subarray
        max_ending_here = max_ending_here + v[i];
        if (max_ending_here > max_so_far)
            max_so_far = max_ending_here;
        if (max_ending_here < 0)
            max_ending_here = 0;
    }
    return max_so_far;
}
int main()
{
    vector<int> a = {1, 2, -4, -5, -6, -7, 8, 9, 0};
    cout << kadanes(a);
    return 0;
}
