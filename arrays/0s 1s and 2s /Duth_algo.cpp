#include <iostream>
using namespace std;
// duth algo
//see the image you will understand everything
//time complexity is n and constant space in place
void sort_array(int arr[], int n)
{
    int low = 0, mid = 0, high = n - 1;
    while (mid <= high)
        switch (arr[mid])
        {
        case 0:
            swap(arr[low++], arr[mid++]);
            break;
        case 1:
            mid++;
            break;
        case 2:
            swap(arr[mid], arr[high--]);
            break;
        }
}
int main()
{
    int arr[] = {2, 1, 1, 0, 0, 2, 1};
    int n = sizeof(arr) / sizeof(arr[0]);
    sort_array(arr, n);
    cout << "Sorted array is:";
    for (int i = 0; i < n; i++)
    {
        cout << " " << arr[i];
    }
    cout << endl;
    return 0;
}