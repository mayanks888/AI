'''Problem 5:

Define a class which has at least two methods:

getString: to get a string from console input
printString: to print the string in upper case.
Also please include simple test function to test the class methods.'''


class demo:
    def getstring(self, str):
        self.mystring = str

    def printstring(self):
        print(self.mystring.upper())


dm = demo()
dm.getstring("mayank")
dm.printstring()
