import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

data=pd.read_csv("../Datasets/census_data.csv")
print (data.head())
print(data.isnull().sum())#sum of all colum null values

#drop purchased column from data sets

#features=features.drop(labels=features.columns[[7]],axis=1)#method one
# features.head()
features=data.iloc[:,0:2].values#method 2
labels=data.iloc[:,-1].values


dt=pd.DataFrame(features)#juct for overhead view convert feature into data frame

label_encode=LabelEncoder()

first_encode=label_encode.fit_transform(features[:,1])
features[:,1]=first_encode
print(len(pd.unique(first_encode)))#inique featues in fist code


#data[:,1].unique()

#print(pd.value_counts(pd.unique(data[:,1])))

one_hot_encode=OneHotEncoder()
first_encode=np.reshape(first_encode,newshape=(len(first_encode),1))
hot_e=one_hot_encode.fit_transform(first_encode)
print(hot_e)

#you have to convert all string column into categorical before applying one hot encoding
sec_hot_code=OneHotEncoder(categorical_features=[1])
features=sec_hot_code.fit_transform(features).toarray()