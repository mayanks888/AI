#code to check if no of closing bracket are equavalent to opening braces

def mismatch_bracket(def_string):
    opening_b=tuple("[{(")
    closing_b=tuple("]})")
    mydict=dict(zip(closing_b,opening_b))
    my_list=[]
    for data in def_string:
        if data in opening_b:
            my_list.append(data)
        elif data in closing_b:
            if len(my_list) ==0:
                return False
            if my_list[-1]==mydict[data]:
                my_list.pop()
            else :
                return False
    return not my_list




if __name__ == '__main__':

    mystr="[[]"

    mystr = "()((()"
    print(mismatch_bracket(mystr))