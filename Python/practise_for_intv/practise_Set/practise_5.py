myDict = {
    "fast": "In a Quick Manner",
    "harry": "A Coder",
    "marks": [1, 2, 5],
    "anotherdict": {'harry': 'Player'},
    1: 2
}
myDict.update({"cool":"icecream"})
myDict["cool"]="icecream"
print(myDict)
# for val in myDict.items():
for val in myDict:
    print(val, myDict[val])
    print()
    # print(myDict[val])