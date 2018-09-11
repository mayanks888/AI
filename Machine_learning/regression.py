
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("C:\Users\mayank\Documents\Python_project\ML\Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


regression_data=LinearRegression()
regression_data.fit(X_train,y_train)
y_predict= regression_data.predict(X_train)
# plt.plot(X,regression_data.predict(X),color='red')
# plt.scatter(X,y)
# plt.show()
print (regression_data.score)

print regression_data.coef_

