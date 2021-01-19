import function_info

while (True):
    # val =int(input("enter int value "))
    try:
        val = int(input("enter int value "))
        if val==5:
            break
        print("squsre of your val is {}".format(int(val)**2))

    except:#custom raise error
        print("something is fishy")
        exit()#this will close the program
    # except Exception as e:
    #     print("{} is not inter value".format(val))
    else:
        print("everythin is cool")
    finally:
        print("last run")


myString="Mayank"
myString=2
# assert isinstance(myString, str)
assert isinstance(myString, str), "%r is not a print value" % myString


