import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt

data= pd.read_csv("dataprepossor.csv")
print np.mean(data['Salary'])
#now checking for outliners
data['Salary']=data[['Salary']].replace(np.nan,np.mean(data['Salary']))
# print data['Salary']
# feature= mydatasets.iloc[:,[1,6]].values
data['Salary'][data['Salary']>4000].count()
# plt=data['Salary'].hist(bins=20)

#___________________________________________________
                    #plot library

# data['salary_log']=np.log(data['Salary'])
# # plt.hist(data['salary_log'],bins=10)
# plt.plot(data['Salary'])
# plt.show()
#___________________________________________________

#we will put the data of datasets inside the eclipse

matrix=np.array([[1,2,1],[4,6,6],[7,8,9],[88,96,97]])
print matrix.shape
outliner =EllipticEnvelope(contamination=.1)
outliner.fit(matrix)
predict_val=outliner.predict(matrix)
print predict_val

#using elispe data on datasets column of age and salary
feature=data.iloc[:,[1,6]].values
print feature
outliner1 =EllipticEnvelope(contamination=.1)
outliner1.fit(feature)
predict_val1=outliner1.predict(feature)
print predict_val1

data['outliner']=predict_val1
print data