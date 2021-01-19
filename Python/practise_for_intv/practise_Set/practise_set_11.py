'''Check if the bracket sequence can be balanced with at most one change in the position of a bracket
Last Updated : 21 Nov, 2019
Given an unbalanced bracket sequence as a string str, the task is to find whether the given string
can be balanced by moving at most one bracket from its original place in the sequence to any other position.
Input: str = “)(()”
Output: Yes
As by moving s[0] to the end will make it valid.
“(())”

Input: str = “()))(()”
Output: No
'''

# Python3 implementation of the approach

# Function that returns true if the sequence
# can be balanced by changing the
# position of at most one bracket
# def canBeBalanced(s, n):
#     # Odd length string can
#     # never be balnced
#     if n % 2 == 1:
#         return False
#
#     # Add '(' in the beginning and ')'
#     # in the end of the string
#     k = "("
#     k = k + s + ")"
#     d = []
#     count = 0
#     for i in range(len(k)):
#
#         # If its an opening bracket then
#         # append it to the temp string
#         if k[i] == "(":
#             d.append("(")
#
#             # If its a closing bracket
#         else:
#
#             # There was an opening bracket
#             # to match it with
#             if len(d) != 0:
#                 d.pop()
#
#                 # No opening bracket to
#             # match it with
#             else:
#                 return False
#
#     # Sequence is balanced
#     if len(d) == 0:
#         return True
#     return False

def find_closing(str):
    opening = tuple("{[(")
    closing = tuple("}])")
    mydict = dict(zip(closing, opening))
    mydict2 = dict(zip(opening, closing))
    new_str = []
    for val in str:
        if val in opening:
            new_str.append(val)
        elif val in closing:
            if not new_str:
                # pass
                new_str.append(val)
            if mydict[val]==new_str[-1]:
                new_str.pop()
            # else:
            #     return "impossible"
    if new_str:
        if len(new_str)!=2:
            return False
        elif new_str[0]==new_str[1] :
            return False
        else:
            return True

    return str

if __name__ == '__main__':

    # Driver code
    S = "()()))"
    n = len(S)
    # if (canBeBalanced(S, n)):
    #     print("Yes")
    # else:
    #     print("No")

    print(find_closing(S))