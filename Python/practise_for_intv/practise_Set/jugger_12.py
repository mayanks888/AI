'''Write a program, which will find all such numbers between
1000 and 3000 (both included) such that each digit of the number
is an even number.
The numbers obtained should be printed in a comma-separated sequence
on a single line.
'''

for val in range(1000, 3001):
    digits = [int(x) for x in str(val) if int(x) % 2 == 0]
    if len(digits) == 4:
        print(val, end=" ,")
