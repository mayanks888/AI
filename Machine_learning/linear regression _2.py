import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# dataset = pd.read_csv('C:\Users\mayank\Documents\Python_project\ML\Startups.csv')
dataset = pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/50_Startups.csv')
X = dataset.iloc[:, 3].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:,1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X=X[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 0)


regression_data=LinearRegression()
regression_data.fit(X_train,y_train)
y_predict= regression_data.predict(X_train)
plt.plot(X,regression_data.predict(X),color='red')
plt.scatter(X,y)
plt.show()

