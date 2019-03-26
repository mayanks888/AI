data='hello world'
dat=data.upper()
print(dat)


# let have some formatting

a=1.235555651546546464
k=34
word="helloWorld"
#converting into binary
print('the value is {:b}'.format(k))
# converting into float

print('the value is {}'.format(a))

print('the value is {:.3f}'.format(a))#upto 3 decimal places


print('the value is {cool:.3f}'.format(cool=a))#to definal value by giving it name first


print('the value is {:11.3f}'.format(a))#here 11 is equavalent ot space or tab ie 11 means 2 tabs
print('the value is {:011.3f}'.format(a))#here 11 is equavalent ot space or tab ie 11 means 2 tabs and reaplace space with 0

# now formatting with sting

print('the string is {cool}'.format(cool=word))#upto 3 decimal places


print('the string is {cool:.2s}'.format(cool=word))#only first 2 words

##########################################################################33333

lsit=[2,44,55,4,5,6,7,8,56,3,54,6,65,4,67,7,5]
k=(sum(lsit))
print(k)
print(lsit[:])
print(lsit[::6])#this is equavalent ot print value with the gap of 6 indexes