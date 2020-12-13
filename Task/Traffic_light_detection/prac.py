a = [2, 9, -10, 5, 18, 9]
print(max(range(len(a)), key=lambda x: a[x]))

k = (max([(v, i) for i, v in enumerate(a)]))
1
