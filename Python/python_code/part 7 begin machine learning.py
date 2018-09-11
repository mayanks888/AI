import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
data= pd.read_csv("dataprepossor.csv")
# print data.head()

features=data
#drop purchased column from data sets

features=features.drop(labels=features.columns[[7]],axis=1)#method one
# features.head()
features=data.iloc[:,:-1].values#method 2
# print features


# now finding labels
labels=data.drop(labels=data.columns[[0,1,2,3,4,5,6]],axis=1)#method one
# print labels
# labels=data.iloc[:,-1].values#method 2
# print labels
# '___________________________________________________________________________'
# #now I will try to fill the non-numerical values with mode values
# mode_data= data["Occupation"].mode()
# print type(mode_data)
# new_data=data["Occupation"].fillna(mode_data.iloc[0])#first method
# # new_data=data["Occupation"].fillna(data["Occupation"].iloc[0])#first method
# print new_data
# '___________________________________________________________________________'

#if we want to fill multiple column with their respective mode then
# defcolumn=['Occupation','Employment Status','Employement Type']
# features[defcolumn]= features[defcolumn].fillna(features.mode().iloc[0])
# print features
# '___________________________________________________________________________'
# label encoder
#this will assign the repeating data to its values

encode=LabelEncoder()
features[:,0]= encode.fit_transform(features[:,0])
features[:,1]= encode.fit_transform(features[:,1])
features[:,2]= encode.fit_transform(features[:,2])
features[:,3]= encode.fit_transform(features[:,3])
features[:,4]= encode.fit_transform(features[:,4])
features[:,5]= encode.fit_transform(features[:,5])
print (features)

hotencode=OneHotEncoder(categorical_features=[1])
features=hotencode.fit_transform(features).toarrary()
print (features)