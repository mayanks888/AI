import pandas as pd
import numpy as np
data=pd.read_csv("/home/mayank-s/Desktop/berkely_filtered.csv")
# data=pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/aptiveBB/reddy.csv')
print(data.head())
print(data.groupby('class').count())

# datasets[(datasets['Pclass']>1) & (datasets['Age']>25)].head()
df=data[(data['xmin']!=data['xmax']) & (data['ymin']!=data['ymax'])]


print(df.head())

df.to_csv("berkely_After_filter.csv")
'''mydata=data.groupby('class')
print(data.groupby('class').count())
index=mydata.groups['car'].values

pyindex=np.random.choice(index, size=10000, replace=False)
data.drop(data.index[pyindex],inplace=True)
print(data.groupby('class').count())

df=data.replace("motor", "cool")
df=df.replace("bike", "cool")
df=df.replace("cool", "motorbike")
df=df.replace("traffic light", "traffic_light")
df=df.replace("traffic sign ", "traffic_sign ")
print(df.groupby('class').count())
data.to_csv("berkely_train_new_1.csv")
print(1)'''