
list =["mayank","Sandy","mishi","shahsank","mayank"]

print(list)
mystirng=""
for val in list:
    # mystirng=mystirng+val+" and "
    mystirng=mystirng.join(val+" and ")
# print(mystirng)


sendtence=" and ".join(list)
print(sendtence)


sec_list=[2,5,6,7,8,9]
mul_list=1

for val in sec_list:
    mul_list*=int(val)
    # return sec_list

print(mul_list)

# lamba sec_list: for val in sec_list, mul_list*=int(val)
sum=[ val*val for val in sec_list]
print(sum)