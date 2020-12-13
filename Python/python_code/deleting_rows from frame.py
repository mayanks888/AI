import pandas as pd

# data=pd.read_csv("/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/berkely_train.csv")
data = pd.read_csv("/home/mayanksati/Documents/csv/BBD_Train_traffic_light.csv")
# data=pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/aptiveBB/reddy.csv')
print(data.head())
mydata = data.groupby('class')
print(data.groupby('class').count())
index = mydata.groups['car'].values

'''pyindex=np.random.choice(index, size=30000, replace=False)
data.drop(data.index[pyindex],inplace=True)
print(data.groupby('class').count())
# data.

classes=['person', 'traffic light', 'bus', 'car', 'motor','bike',  'traffic sign']
# df=data.loc[data['class'].isin(["traffic sign","traffic light"])]
df=data.loc[data['class'].isin(classes)]
print(df.groupby('class').count())'''

# df.to_csv("berkely.csv")

'''mydata=df.groupby('class')
index=mydata.groups['car'].values
# mydata.groups['car']._data

pyindex=np.random.choice(index, size=30000, replace=False)
df.drop(df.index[pyindex],inplace=True)
print(df.groupby('class').count())'''

# df = pd.DataFrame.drop('car',axis=1,columns='class')
# df = pd.DataFrame.drop('car')
# df.drop(labels='car',axis=1)
# print(df.groupby('class').count())
# cool=df['class'=='car']

# df = pd.DataFrame(np.random.randn(50, 4), columns='class'))


import pandas as pd
import numpy as np
data=pd.read_csv("/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/berkely_train.csv")
# data=pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/aptiveBB/reddy.csv')
print(data.head())
mydata=data.groupby('class')
print(data.groupby('class').count())
index=mydata.groups['car'].values

pyindex=np.random.choice(index, size=30000, replace=False)
data.drop(data.index[pyindex],inplace=True)

# data.
cool=data.iloc[:,1:7].values
classes=['person', 'traffic light', 'bus', 'car', 'motor','bike',  'traffic sign']
# df=data.loc[data['class'].isin(["traffic sign","traffic light"])]
df=data.loc[data['class'].isin(classes)]
print(df.groupby('class').count())

# df = pd.DataFrame.drop('car',axis=1,columns='class')
# df = pd.DataFrame.drop('car')
df.drop(labels='car',axis=1)
print(df.groupby('class').count())
# cool=df['class'=='car']

# df = pd.DataFrame(np.random.randn(50, 4), columns='class'))'''
