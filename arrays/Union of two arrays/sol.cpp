/*
qn: https://practice.geeksforgeeks.org/problems/union-of-two-arrays3538/1
Given two arrays a[] and b[] of size n and m respectively. The task is to find the number of elements in the union between these two arrays.

Union of the two arrays can be defined as the set containing distinct elements from both the arrays. If there are repetitions, then only one occurrence of element should be printed in the union.

Note : Elements are not necessarily distinct.

my approacj:
first sort both then two pointer and traverse
plus we need the count of unions
will it suffice?
lets see

omg omg omg union of two arrays is really good one what we do
 is basically sort both then we do two pointer blah blah blah....
better way is using unordered_set<int> and just keep inserting
 it, it will just keep it once in it lmao this is too good,
  but some may not like it so normal way is also good O((m+n)log(m+n))
   and the set way is O(m+n) so yeah optimising way its really good
but the issue is space of O(m+n) so two pointer if we need
 to save space and set if we want to save time


*/
