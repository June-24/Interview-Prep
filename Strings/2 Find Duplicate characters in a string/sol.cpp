/*
qn: https://www.codingninjas.com/codestudio/problems/duplicate-characters_3189116?topList=love-babbar-dsa-sheet-problems&utm_source=website&utm_medium=affiliate&utm_campaign=450dsatracker&leftPanelTab=1

*/
#include <bits/stdc++.h> 
vector<pair<char,int>> duplicate_char(string s, int n){
    // Write your code here.
    map<char,int> mp;
    for(int i=0;i<s.length();i++){
        mp[s[i]]++;
    }
    vector<pair<char,int>> ans;
    for(auto i:mp){
        if(i.second>1){
            ans.push_back(make_pair(i.first,i.second));
        }
    }
    return ans;
}