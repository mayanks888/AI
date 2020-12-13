import pandas as pd
import numpy as np
# data=pd.read_csv("/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/berkely_train.csv")
data = pd.read_csv("/home/mayank_s/Desktop/template/gstreet/final_gstreet.csv")

# data=pd.read_csv("/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/berkely_train.csv")
# data=pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/aptiveBB/reddy.csv')
print(data.head())
# mydata=data.groupby('img_name')
# print(data.groupby('img_name').count())
# index=mydata.groups['car'].values

# pyindex=np.random.choice(index, size=30000, replace=False)
# data.drop(data.index[pyindex],inplace=True)

# data.
cool=data.iloc[:,0:12].values
classes=[31,54,1,53,68,10]
# classes=['person', 'traffic light', 'bus', 'car', 'motor','bike',  'traffic sign']
# df=data.loc[data['class'].isin(["traffic sign","traffic light"])]
df=data.loc[data['boxno'].isin(classes)]
print(df.groupby('class').count())

# df = pd.DataFrame.drop('car',axis=1,columns='class')
# df = pd.DataFrame.drop('car')
df.drop(labels='car',axis=1)
print(df.groupby('class').count())
# cool=df['class'=='car']

# df = pd.DataFrame(np.random.randn(50, 4), columns='class'))'''
