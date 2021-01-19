
a=34
def check_global():
    a=3
    global a

    print(a)

    print(a)

check_global()