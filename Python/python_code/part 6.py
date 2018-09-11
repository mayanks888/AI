import pandas as pd
import numpy as np

data= pd.read_csv("dataprepossor.csv")
print data.head()

#get rows that contains null
print data.isnull().sum()

#what if you want data column without any empty  value
print data.dropna(axis =1)

#you can also find all the rows without missing value
print data.dropna(axis=0)

print data.info()

#print particular rows
print data.loc[1]
# features=

features=data
#drop purchased column from data sets

features=features.drop(labels=features.columns[[7]],axis=1)#method one
features.head()
features=data.iloc[:,:-1].values#method 2
print features


# now finding labels
labels=data.drop(labels=data.columns[[0,1,2,3,4,5,6]],axis=1)#method one
print labels
labels=data.iloc[:,-1].values#method 2
print labels