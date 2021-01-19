class myEmployee():
    def __init__(self):
        dog = 1
        cat = 3
        print("constructor ran")

    def addVal(self, a, b):
        return (a + b)


obj = myEmployee()
print(obj.addVal(2, 3))
