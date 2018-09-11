import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
import statsmodels as st

# '______________________________________________________________' \
data=[1,2,3,4,5,6,7,8,9]
data_ary=np.array(data)
# print np.mean(data)
# print np.median(data)
# print stats.mode(data)
# print np.percentile(data,25)
# print np.percentile(data,75)
# print np.var(data)
# print np.std(data)
# '______________________________________________________________' \
# linear algebra

matrix=np.array([[1,2,1],[4,0,6],[7,8,9],[88,96,97]])

# print matrix
# print np.mean(matrix)
# print np.var(matrix)
# print stats.mode(matrix,axis=1)
# # '______________________________________________________________' \
                            #scaling of datas

                            #min max scaling
# minmaxscale=preprocessing.MinMaxScaler(feature_range=(0,1))
# scale_data=minmaxscale.fit_transform(matrix)
# print scale_data

                            #standard scaling
# this is a standard scaler=[(Xi-Xmean)/standard deviation]

standardscaler=preprocessing.StandardScaler()
x_scale=standardscaler.fit_transform(matrix)
print x_scale

# # '______________________________________________________________' \
        # Here I did scaling to age and salary of datasets
# mydatasets= pd.read_csv("dataprepossor.csv")
# # datasets['Sex']=datasets['Sex'].replace('male','men').head()
# mydatasets[['Salary','Age']]=mydatasets[['Salary','Age']].replace(np.nan,0)
# feature= mydatasets.iloc[:,[1,6]].values
# standardscaler2=preprocessing.StandardScaler(0,2)
# x_scale=standardscaler2.fit_transform(feature)
# print x_scale
#
#
# feature1= mydatasets["Age"]
# feature1matrix=feature1.values.reshape(-1,1)
#
# # scaling my data based on min max scale
# minmaxscale=preprocessing.MinMaxScaler(feature_range=(0,10))
# x_scale=minmaxscale.fit_transform(feature1matrix)
#
# # scaling my data based on standard scaling
# standardscaler2=preprocessing.StandardScaler(0,10)
# # x_scale=standardscaler2.fit_transform(feature1matrix)
# print x_scale


# # '______________________________________________________________' \
            #normaliser xi/sqrt(sum of all square of rows

normaliser1=preprocessing.Normalizer()
normaldata=normaliser1.fit_transform(matrix)
print normaldata


# # '______________________________________________________________' \
                                # Binarizer

binarizer=preprocessing.Binarizer(4)#here 4 means that all value below 4 will be 0 and above will  be 1
binardata=binarizer.fit_transform(matrix)
print binardata
