/*
qn:https://leetcode.com/problems/find-the-duplicate-number/
i think that XOR will work out
no extra space
explanation: https://leetcode.com/problems/find-the-duplicate-number/solutions/1892872/c-algorithm-4-approaches-binary-search-brute-force-cnt-array-map/?orderBy=most_votes
explanation for fast pointer slow pointer approach :https://leetcode.com/problems/find-the-duplicate-number/solutions/72846/my-easy-understood-solution-with-o-n-time-and-o-1-space-without-modifying-the-array-with-clear-explanation/?orderBy=most_votes

nopes
floyds cycle detection

One possible approach to solve
this problem is to use Floyd's
 Tortoise and Hare (Cycle Detection) algorithm.

Think of the array as a
 linked list where each element points
  to the index of the next element (array
   value is the index). In this case, the
    repeated number will be the intersection
     point of the cycle.

Use two pointers (tortoise and hare)
 that start at the same position. Move
  the tortoise one step and the hare two steps
   at a time. If there is a cycle in the linked
   list, eventually the hare will catch up to the
    tortoise at some point. Once the hare and tortoise
     meet, move one of them to the beginning of the array
     and move both pointers one step at a time.
     The intersection point (repeated number) will be
     found when both pointers point to the same element.

This algorithm has a time complexity of O(n)
and a space complexity of O(1), satisfying the
requirements of the problem.


*/
#include <bits/stdc++.h>
using namespace std;
class Solution
{
public:
    int findDuplicate(vector<int> &nums)
    {
        int tortoise = nums[0];
        int hare = nums[0];
        do
        {
            tortoise = nums[tortoise];
            hare = nums[nums[hare]];
        } while (tortoise != hare);
        tortoise = nums[0];
        while (tortoise != hare)
        {
            tortoise = nums[tortoise];
            hare = nums[hare];
        }
        return tortoise;
    }
};
int main()
{

    return 0;
}

// class Solution {
// public:
//     int findDuplicate(vector<int>& nums) {
//         // Initialize the tortoise and hare pointers
//         int tortoise = nums[0];
//         int hare = nums[0];

//         // Move the tortoise and hare pointers until they meet
//         do {
//             tortoise = nums[tortoise];
//             hare = nums[nums[hare]];
//         } while (tortoise != hare);

//         // Move one of the pointers to the beginning of the array
//         tortoise = nums[0];

//         // Move both pointers one step at a time until they meet at the intersection point
//         while (tortoise != hare) {
//             tortoise = nums[tortoise];
//             hare = nums[hare];
//         }

//         // Return the intersection point (repeated number)
//         return tortoise;
//     }
// };
