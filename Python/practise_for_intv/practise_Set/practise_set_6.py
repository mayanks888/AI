name="hellomayank"
if len(name)>10:
    print("hurrY")

mylist=["hello","mayank","clz"]
print(mylist.count("hello"))
print(mylist.index("mayank"))

#search for valueof list that contain cetein values
out_list=[val for val in mylist if "a" in val]
print(out_list)