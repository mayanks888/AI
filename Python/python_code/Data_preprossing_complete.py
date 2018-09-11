import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

data=pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/Data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,3].values
#########################################################################################
#firt step should be find any a any empy item in data sets
#so the best method is to replce the empty item eith the mean of a the particular features

# TAking care of missing data here i'll replace all mean value with the mean values
# and also explain the diff bw fit and transform
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
# imputer=imputer.fit(x[:,1:3])
# x[:,1:3]=imputer.transform(x[:,1:3])
                # or fit and replave in one line only
x[:,1:3]=imputer.fit_transform(x[:,1:3])
print(x)
#########################################################################################
#next job is to work on         categorical varaibles

#first one is coveting catergorical text into the integer label
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

label_encode=LabelEncoder()
first_encode=label_encode.fit_transform(x[:,0])
x[:,0]=first_encode
print(len(pd.unique(first_encode)))#inique featues in fist code
print(first_encode)
print(x)

#########################################################################################
#       but still the values are ini integer so next job should be to convert into one hot encoder

one_hot_encode=OneHotEncoder(categorical_features=[0])#categorical_features=[0]) since we have one hot encode at index 0
x=one_hot_encode.fit_transform(x).toarray()#to array is imp as it convert it into aray of one hot encode
print (x)

#########################################################################################
#                now label encode y(labels)
label_encode=LabelEncoder()
y=label_encode.fit_transform(y)
print (y)
#########################################################################################
#                   now splitting data in traina nd test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 0)
#########################################################################################
#                   next is feature scalilng
#Here I have standardise the features (set the value between mean 0 and sd=1))
# this is been done so that one feature should not dominate other freatures (very important)
from sklearn.preprocessing import StandardScaler
scale_val=StandardScaler()
scale_x=scale_val.fit_transform(X_train)
# we dont have to do fit_transfrom for x_test best its already fit in above step
# One more imp question to discuss is weather one hot encode feature should also be scaled
scale_c_test = scale_val.transform(X_test)
print("the scaler mean  is",scale_x[0].mean(),scale_x[1].mean())
print(scale_x)