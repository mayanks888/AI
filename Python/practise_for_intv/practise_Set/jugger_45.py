'''Define a class named American which has a static method called printNationality.'''
# statci mithod can be c/all without creating the object of thr class
class demo:
    def __init__(self):
        print("construtor")

    @staticmethod
    def printsomething():
        print("hello")

class servant(demo):
    def __init__(self,demo):
        demo.printsomething()



dm=demo()

print(demo.printsomething())