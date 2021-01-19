import os

filepath = "/home/mayank_s/saved_work/AI_github/AI/python_tensorflow.txt"
if os.path.exists(filepath):
    with open(filepath) as f:
        for i, line in enumerate(f):
            print(line)

#####################33
# more on os
print(os.path.join("cool", "bool"))
