# Problem 1:
#
# Write a program which will find all such numbers which are
# divisible by 7 but are not a multiple of 5, between 2000 and 3200
# (both included).The numbers obtained should be printed in a
# comma-separated sequence on a single line.

mylist=[val for val in range(2000,3200) if (val%7==0 and val/5!=0)]
print(mylist)
print("".join(str(mylist)))
for val in mylist:
    print(str(val),end=",")


# list =["mayank","Sandy","mishi","shahsank","mayank"]
# sendtence=" and ".join(list)
# print(sendtence)