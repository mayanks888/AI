'''Problem 57:

Write a program to read an ASCII string and to convert it to a unicode
string encoded by utf-8.
'''

val="k"

# print(ascii(val))
val.encode()
u = val.encode('utf-8')
u = ord(val)
print(u)

def f(n):
    if n < 2:
        return n
    return f(n-1) + f(n-2)

n = int(input())
print(f(n))