import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smapi


d
data=pd.read_excel("EngRpm.XLS")

new_data=data[['Gr[]','WhlRPM_FL[rpm]','EngRPM[rpm]','AccelPdlPosn[%]','EngTrq[Nm]','EngRPM[rpm]']]#to print new column of your choice
data.dropna(inplace=False)



# print new_data[(new_data['Gr[]'].notnull())|(new_data['WhlRPM_FL[rpm]'].notnull())].head(500)
data= new_data[(new_data['WhlRPM_FL[rpm]'].notnull())]
# print datasets[(datasets['Pclass']>1) & (datasets['Age']>25)].head()
# print new_data ['Gr[]'].isnull().sum()
# print new_data ['WhlRPM_FL[rpm]'].isnull().sum()
# print data['WhlRPM_FL[rpm]'].notnull().sum()
# print data['Gr[]'].notnull().sum()


means=df1.mean()
print means
print"covariance is"
print df1.cov()
print 'variance is '
print np.var(df1)
print 'correlation is '
print df1.corr(method='pearson')
# plt.scatter(df1['Miles Travel'],df1['Travel time'])
# plt.scatter(df1['Gas price'],df1['Travel time'])
# plt.scatter(df1['Miles Travel'],df1['Num Deliveries'])
# plt.show()

# _________________________________________

features=df1.iloc[:,:-1].values
labels=df1.iloc[:,3].values

print features
print labels


x_train,xtest,y_train,ytest=train_test_split(features,labels,test_size=1,random_state=0)
#
# ?_______________________________________________
# now will call lineat regression function to find the minimum optimal line in the train datasets
regresion_data=LinearRegression()
regresion_data.fit(x_train,y_train)

print regresion_data.coef_#this is the slope of optimam line
print regresion_data.intercept_# this is y intercept


plt.scatter(x_train,y_train,color='b')
plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.show()

#lets test the accuracy of our equation line
print 'train accuracy ', regresion_data.score(x_train,y_train)
print 'test accuracy ', regresion_data.score(xtest,ytest)
# print 'r_square',regre
# ?_______________________________________________
# #this is used for predicting all  values from the regression that we found from regression function
exp_Data=labels
exp_Data=labels.reshape(-1,1)

