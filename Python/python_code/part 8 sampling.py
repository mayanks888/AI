import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
data= pd.read_csv("dataprepossor.csv")
# print data.head()
# print data.head()

features=data
#drop purchased column from data sets

features=features.drop(labels=features.columns[[7]],axis=1)#method one
print features.head()
# features=data.iloc[:,:-1].values#method 2
# print features


# now finding labels
labels=data.drop(labels=data.columns[[0,1,2,3,4,5,6]],axis=1)#method one
print labels
# labels=data.iloc[:,-1].values#method 2
# print labels


# if we want to fill multiple column with their respective mode then
defcolumn=['Occupation','Employment Status','Employement Type']
features[defcolumn]= features[defcolumn].fillna(features.mode().iloc[0])
# print features


# this will divide your training  datasets and test  datasets by 75% and 25%
x_train,xtest,ytrain,ytest=train_test_split(features,labels,test_size=.25,random_state=0)
print x_train
print ytrain
print xtest
print x_train.shape
print xtest.shape
