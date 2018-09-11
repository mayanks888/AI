import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
print (dataset['State'].unique)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
print (X)
X=X[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 0)


regression_data=LinearRegression()
regression_data.fit(X_train,y_train)
y_predict= regression_data.predict(X_train)

'''print (regression_data.intercept_)
print (regression_data.coef_)
import statsmodels.formula.api as sm
X=np.append(np.ones((50,1)).astype(int),values=X,axis=1)
# print X
# mymodel=X[:,[0,1,2,3,4,5]]#previous model data whole
mymodel=X[:,[0,3]]
reg_summary=sm.OLS(endog=y,exog=mymodel).fit()
'
print (reg_summary.summary())'''

