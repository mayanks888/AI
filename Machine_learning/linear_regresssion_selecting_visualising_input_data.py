import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/housing_data.csv')#,header=None)#, sep='\s+')
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(data.head())


# after load we have to do exploratory data analysis
# Using this scatterplot matrix, we can now quickly eyeball how the data is distributed and whether it contains outliers.
'''
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
# [ 282 ]Chapter 10
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(data[cols], size=2.5)
#or lets plot everything
# sns.pairplot(data, size=2.5)
# plt.show()

# ************************************************************************
# let check corelation between the data
import numpy as np
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
# plt.show()
# data.to_csv('housing_data.csv')'''

# ************************************************************************
X = data[['RM']].values
y = data['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 0)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
sc_x = StandardScaler()
sc_y = StandardScaler()
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# X_std = sc_x.fit_transform(X)
# y_std = sc_y.transform(y)
'''lr = LinearRegression()
lr.fit(X_std, y_std)
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()'''

regression_data=LinearRegression()
regression_data.fit(X_train,y_train)
y_train_pred= regression_data.predict(X_train)
y_test_pred= regression_data.predict(X_test)
plt.scatter(X,y,color='blue')
plt.plot(X_train,regression_data.predict(X_train),color='red')
plt.scatter(X,y)
plt.show()

from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

from sklearn.metrics import r2_score
rscore=r2_score(y_test,y_test_pred)
print(rscore)