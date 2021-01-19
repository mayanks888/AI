'''Problem 8:

Write a program that accepts a comma separated sequence of words
as input and prints the words in a comma-separated sequence after
sorting them alphabetically.

Suppose the following input is supplied to the program:

without,hello,bag,world
Then, the output should be:

bag,hello,without,world'''

# val=input()
val = 'without hello bag world'
mylist = val.split(" ")
mylist.sort()
print(mylist)

str = "Hello world Practice makes perfect"
print(str.upper())

str = "hello world and practice makes perfect and hello world again"
str_lsit = str.split(" ")
new_str = [val.upper() for val in str_lsit]
cool = (" ".join(set(new_str)))
print(cool)
#
# # or
# word = input().split()
#
# for i in word:
#     if word.count(i) > 1:    #count function returns total repeatation of an element that is send as argument
#         word.remove(i)     # removes exactly one element per call
#
# word.sort()
# print(" ".join(word))


st = "hello World COOL"
mystr = [val for val in st if val.islower()]
print(f"total lower cases are {len(mystr)} and tolal upper caser are {(len(st) - len(mystr))}")
