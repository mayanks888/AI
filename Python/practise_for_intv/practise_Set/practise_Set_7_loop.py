        # *
        # **
        # ***
        # ****

piramid_height=20
fist_space=int(piramid_height/2)
for a in range(piramid_height):
    if not a%2==0:
        fist_space = fist_space - 1
        for k in range(fist_space):
            print(" ", end="")
        for b in range(a):
            print("*",end="")
        print()

#####################################333

print("\n")
#reverse printing
piramid_height=20
fist_space=0
for a in range(piramid_height,-1,-1):
    if not a%2==0:
        fist_space = fist_space + 1
        for k in range(fist_space):
            print(" ", end="")
        for b in range(a):
            print("*",end="")
        print()



##############################################33

# # #
#   #
# # #

limit=9
mid_range=int(limit/2)
for a in range(limit):
    for b in range(limit):
        if a==mid_range and b==mid_range:
            print(" ", end=" ")
        else:
        # print(a,b)
            print("*",end=" ")
    print()