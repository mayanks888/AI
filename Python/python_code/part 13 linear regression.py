import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

data= pd.read_csv("Salary_Data.csv")
# print data

# plt.scatter(data['YearsExperience'],data['Salary'])
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')

# plt.show()

# ?_______________________________________________
# divide data into train datasets and test datasets


features=data.iloc[:,:1].values
labels=data.iloc[:,1].values

# print features

x_train,xtest,y_train,ytest=train_test_split(features,labels,test_size=.25,random_state=0)
# print x_train
# print y_train
# print xtest
# print x_train.shape
# print xtest.shape

# ?_______________________________________________
# now will call lineat regression function to find the minimum optimal line in the train datasets
regresion_data=LinearRegression()


regresion_data.fit(x_train,y_train)

print regresion_data.coef_#this is the slope of optimam line
print regresion_data.intercept_# this is y intercept


plt.scatter(x_train,y_train,color='b')
plt.plot(x_train,regresion_data.predict(x_train),color='r')
# plt.show()

#lets test the accuracy of our equation line
print 'train accuracy ', regresion_data.score(x_train,y_train)
print 'test accuracy ', regresion_data.score(xtest,ytest)
# print 'r_square',regre
# ?_______________________________________________
#this is used for predicting all  values from the regression that we found from regression function
exp_Data=data['YearsExperience']
exp_Data=data['YearsExperience'].reshape(-1,1)
print exp_Data.shape
data['predicted value']= regresion_data.predict(exp_Data)
print data

# ?_______________________________________________
'''now we will find the following
RSS - Residual sum of squares
MSE - Mean Squared Error
RMSE - Root Mean Squared Error
RSS = ((y_test - y_predtest)**2).sum()
MSE = np.mean((y_test - y_predtest)**2)
RMSE = np.sqrt(MSE)'''

rsqaure_value=r2_score(data['Salary'],regresion_data.predict(exp_Data))
print 'the r square value is ',rsqaure_value
print('the r square value is  or Variance score: %.2f' % rsqaure_value)
print 'mean square value', mean_squared_error(data['Salary'],regresion_data.predict(exp_Data))