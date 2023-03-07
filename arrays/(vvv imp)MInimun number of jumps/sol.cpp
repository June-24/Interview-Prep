/*
qn:https://practice.geeksforgeeks.org/problems/minimum-number-of-jumps-1587115620/1


vvvvv important code man
greedy can do in n time

good explanation: https://www.youtube.com/watch?v=_6QpiqTw_ew

*/
//{ Driver Code Starts
#include <bits/stdc++.h>
using namespace std;

// } Driver Code Ends
// Function to return minimum number of jumps to end of array

class Solution
{
public:
    int minJumps(int arr[], int n)
    {
        // // Your code here
        // my n^2 version not working
        // int i = 0;
        // int moves = 0;
        // while (i < n - 1)
        // {
        //     if (i == 0 && arr[0] == 0)
        //         return -1;
        //     int max = INT_MIN;
        //     int index = -1;
        //     for (int j = i + 1; j <= i + arr[i]; j++)
        //     {
        //         if (max < arr[j])
        //             max = arr[j], index = j;
        //     }
        //     if (max == 0)
        //     {
        //         return -1;
        //     }
        //     i += index;
        //     moves++;
        // }
        // return moves;

        // som eoptimised version in n complexity
        int steps = arr[0];
        int maxReach = arr[0];
        int jumps = 1;
        if (arr[0] == 0)
            return -1;
        if (n <= 1)
            return 0;
        for (int i = 1; i < n; i++)
        {
            if (i == n - 1)
                return jumps;
            maxReach = max(maxReach, i + arr[i]);
            steps--;
            if (!steps)
            {
                // ran out of steps so need to make a jump
                jumps++;
                if (i >= maxReach)
                    return -1;
                steps = maxReach - i;
            }
        }
        return -1;
    }
};

//{ Driver Code Starts.

int main()
{
    int t;
    cin >> t;
    while (t--)
    {
        int n, i, j;
        cin >> n;
        int arr[n];
        for (int i = 0; i < n; i++)
            cin >> arr[i];
        Solution obj;
        cout << obj.minJumps(arr, n) << endl;
    }
    return 0;
}

// } Driver Code Ends
