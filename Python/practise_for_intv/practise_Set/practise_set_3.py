myletter = '''hello dude how is daughter doing'''

myletter = myletter.replace("dude", "mayank")
myletter = myletter.replace("daughter", "mishika")
print(myletter)

# find and replace double space

new_string="hello         my name is mayank  sati"
# print(new_string.find("\t"))
print(new_string.count(" +"))
print(new_string)

# old method of removing extra white space
new_string=new_string.replace(" +", " ")
print(new_string)
mysplit=new_string.split(" ")
str=[val for val in mysplit if val !=""]

print(" ".join(str))

#new ways to remove white space
# import re
# re.sub(' +', ' ', 'The     quick brown    fox')
# 'The quick brown fox'
# print(re.sub(' +', ' ',new_string))
string_2="mayank"
# cool=string_2.split()
print(list(string_2))
