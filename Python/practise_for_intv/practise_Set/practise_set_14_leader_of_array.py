'''Given an array A of positive integers. Your task is to find the leaders in the array.
An element of array is leader if it is greater than or equal to all the elements to its right side.
The rightmost element is always a leader.

Input:
N = 6
A[] = {16,17,4,3,5,2}
Output: 17 5 2
Explanation: The first leader is 17
as it is greater than all the elements
to its right.  Similarly, the next
leader is 5. The right most element
is always a leader so it is also
included.
 '''


def find_leader_of_array(myarray):
    for i in range(len(myarray)):
        for j in range(i , len(myarray)):
            # print(myarray[i],myarray[j])
            if myarray[i] < myarray[j]:
                break
        if j == len(myarray)-1:
            print(myarray[i], end=" ")


def printLeaders(arr):
    size=len(arr)
    for i in range(0, size):
        flag = True
        for j in range(i + 1, size):
            if arr[i] < arr[j]:
                flag=False
                break

        if flag:  # If loop didn't break
            print(arr[i],end=" ")


if __name__ == '__main__':
    arr = [16, 17, 4, 3, 5,1,2]
    find_leader_of_array(arr)
    print('\n')
    printLeaders(arr)
