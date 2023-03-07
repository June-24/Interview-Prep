/*
some other way to do the sam ething
i think the first way is simpler
*/
#include <iostream>
#include <climits>

using namespace std;

int kadanes_algorithm(int arr[], int n) {
    int max_so_far = arr[0];
    int max_ending_here = arr[0];

    for (int i = 1; i < n; i++) {
        max_ending_here = max(arr[i], max_ending_here + arr[i]);
        max_so_far = max(max_so_far, max_ending_here);
    }

    return max_so_far;
}

int main() {
    int arr[] = {-2, -3, 4, -1, -2, 1, 5, -3};
    int n = sizeof(arr)/sizeof(arr[0]);
    int max_sum = kadanes_algorithm(arr, n);

    cout << "Maximum subarray sum is " << max_sum << endl;

    return 0;
}
