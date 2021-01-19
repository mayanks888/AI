'''Given an incomplete bracket sequence S. The task is to find the number of closing brackets ‘)’ needed to make it a regular bracket sequence and print the complete bracket sequence. You are allowed to add the brackets only at the end of the given bracket sequence. If it is not possible to complete the bracket sequence, print “IMPOSSIBLE”.

Let us define a regular bracket sequence in the following way:

Empty string is a regular bracket sequence.
If s is a regular bracket sequence, then (s) is a regular bracket sequence.
If s & t are regular bracket sequences, then st is a regular bracket sequence.

Input : str = “(()(()(”
Output : (()(()()))
Explanation : The minimum number of ) needed to make the sequence regular are 3 which are appended at the end.

Input : str = “())(()”
Output : IMPOSSIBLE
'''

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
            if mydict[val]==new_str[-1]:
                new_str.pop()
            else:
                return "impossible"
    if new_str:
        for dat in new_str:
            str=str+mydict2[dat]
    print(new_str)
    return str


def completeSequence(s):
    # Finding the length of sequence
    n = len(s)

    open = 0
    close = 0

    for i in range(n):

        # Counting opening brackets
        if (s[i] == '('):
            open += 1
        else:

            # Counting closing brackets
            close += 1

        # Checking if at any position the
        # number of closing bracket
        # is more then answer is impossible
        if (close > open):
            print("IMPOSSIBLE")
            return

    # If possible, print 's' and
    # required closing brackets.
    print(s, end="")
    for i in range(open - close):
        print(")", end="")

if __name__ == '__main__':
    mystr="(()(()("
    # print(find_closing(mystr))
    (completeSequence(mystr))

    ")()()("