
# inputVar=int(input())#user input
inputVar=4
print(inputVar)
if inputVar>2:
    print("cool")
elif inputVar<=2:
    print("Fool5")


a=2
b=-2

print("check logical and comapratr ",a >0 and b <0)

myString="Mayank"
cool=myString.center()
# myString=2
# assert isinstance(myString, str)
assert isinstance(myString, str), "%r is not a print value" % myString
print( "get value of strint","m" in myString)
if "M "in myString:
    print (True)
else:
    print(False)