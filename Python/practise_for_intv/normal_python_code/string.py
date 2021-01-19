#sting are imutable ie nother can chage the value of string

my_string="hello_world"
print(my_string[0])
# my_string[0]="k" #non mutable
strip=my_string.strip("")
print(strip)
my_string.upper()

#######################3
# list to string

my_list=["this", "is", "it"]
str="  ".join(my_list)
print(str)

# *************************
#best way to print or add string
name,place="mayank","dehradun"
mytring=f"hello my name is {name} and i live in {place}"
print(mytring)