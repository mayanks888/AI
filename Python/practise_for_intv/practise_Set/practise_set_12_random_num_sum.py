'''Finding sum of digits of a number until sum becomes single digit
Difficulty Level : Medium
Given a number n, we need to find the sum of its digits such that:

If n < 10
    digSum(n) = n
Else
    digSum(n) = Sum(digSum(n)
    Input : 1234
Output : 1
Explanation : The sum of 1+2+3+4 = 10,
              digSum(x) == 10
              Hence ans will be 1+0 = 1
'''

from functools import reduce

input_number = 1234689
sum = 0
# while (input_number > 0):
#     sum += input_number % 10
#     input_number = int(input_number / 10)
# input_number = 1234567874
func = lambda a, b: a + b
# mylist=[1,2,3,4]
# mytupel=list(tuple("1234"))
# print(mytupel)
create_list = [int(i) for i in str(input_number)]
while (True):
    val = reduce(func, create_list)
    print(val)
    new_list = [int(i) for i in str(val)]
    if len(new_list) == 1:
        break
    else:
        create_list = new_list

print(val)

#################################################33
def digSum( n):
    sum = 0

    while(n > 0 or sum > 9):

        if(n == 0):
            n = sum
            sum = 0

        sum += n % 10
        n =int(n/ 10)

    return sum

print(digSum(input_number))