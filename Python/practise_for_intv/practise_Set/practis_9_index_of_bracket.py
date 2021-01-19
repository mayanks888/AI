'''Given a string with brackets. If the start index of the open bracket is given, find the index of the closing bracket.

Examples:

Input : string = [ABC[23]][89]
        index = 0
Output : 8
The opening bracket at index 0 corresponds
to closing bracket at index 8.'''

def find_index(mystr,indx):
    flag=False
    opening=tuple("{[(")
    closing=tuple("}])")
    print(opening)
    mydict=dict(zip(closing,opening))
    first=[]
    end_=[]
    second_dict={}
    for i,data in enumerate(mystr):
        print(data)
        if i==indx:
            flag=True
        if data in opening:
            first.append(i)
            # second_dict[i]=i
        elif data in closing:
            end_.append(i)
    mydict = dict(zip(first, end_))


    print(mydict)


# Python program to find index of closing
# bracket for a given opening bracket.
from collections import deque


def getIndex(s, i):
    # If input is invalid.
    if s[i] != '[':
        return -1

    # Create a deque to use it as a stack.
    d = deque()

    # Traverse through all elements
    # starting from i.
    for k in range(i, len(s)):

        # Pop a starting bracket
        # for every closing bracket
        if s[k] == ']':
            d.popleft()

            # Push all starting brackets
        elif s[k] == '[':
            d.append(s[i])

            # If deque becomes empty
        if not d:
            print(k)
            return k


    return -1


if __name__ == '__main__':
    # str="[ABC[2[3]]][{89}]"
    str="[ABC[2][3]5]5]"
    idex=0
    # find_index(str,idex)
    getIndex(str,idex)