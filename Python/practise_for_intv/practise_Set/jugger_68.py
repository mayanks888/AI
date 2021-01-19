'''Problem 66:

Please write a program which accepts basic mathematic expression from console and print the evaluation result.

Example: If the following n is given as input to the program:

35 + 3
Then, the output of the program should be:

38'''

expression = input()
# expression=32+3
ans = eval(expression)
print(ans)