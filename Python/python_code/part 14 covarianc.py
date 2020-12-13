import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df1 = pd.DataFrame()
X1 = [12, 30, 15, 24, 14, 18, 28, 26, 19, 27]
Y1 = [20, 60, 27, 50, 21, 30, 61, 54, 32, 57]
Z1 = [1, 56, 6, 72, 66, 95, 8, 67, 42, 17]
df1['X'] = X1
df1['Y'] = Y1
df1['Z'] = Z1
print(df1)

means = df1.mean()
print(means)
print("covariance is")
print(df1.cov())
print('variance is ')
print(np.var(df1))
print('correlation is ')
print(df1.corr(method='pearson'))
plt.scatter(df1['X'], df1['Y'])
plt.show()
