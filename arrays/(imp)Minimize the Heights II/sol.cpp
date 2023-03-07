/*
qn: https://practice.geeksforgeeks.org/problems/minimize-the-heights3351/1
logic given in the enclosed image
//man this is freaking important
vvvvvv important
best approach: https://www.youtube.com/watch?v=Av7vSnPSCtw
yeaaah
*/
#include <bits/stdc++.h>
using namespace std;
int minimize(int arr[], int n, int k)
{
    //below approach not working

    // sort(arr, arr + n);
    // int ans = arr[n - 1] - arr[0];
    // int small = arr[0] + k;
    // int big = arr[n - 1] - k;
    // if (small > big)
    //     swap(big, small);
    // for (int i = 1; i < n; i++)
    // {
    //     int subtract = arr[i] - k;
    //     int add = arr[i] + k;
    //     if (subtract >= small && add <= big)
    //         continue;
    //     else if (subtract < small)
    //         small = subtract;
    //     else
    //         big = add;
    // }
    // return min(ans, big - small);

    
    //better approach
    sort(arr,arr+n);
    int ans=arr[n-1]-arr[0];
    int small=arr[0]+k;
    int big=arr[n-1]-k;
    int Max,Min;
    for(int i=0;i<n-1;i++){
        Min=min(small,arr[i+1]-k);
        Max=max(big,arr[i]+k);
        if(Min<0)continue;
        ans=min(ans,Max-Min);
    }
    return ans;
}

int main()
{

    return 0;
}
