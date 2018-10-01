import pandas as pd
import numpy as np

'''data=pd.read_csv('Carcount.csv')
data=data.sort_values(by='count', ascending=False)
print(data)

df=data[data['count']>12]
print(df)
# df.to_csv('maxcarcaount_3.csv')
# df['xmax']-df['xmin'] > threshold
# df[df['JobTitle'].value_counts()<2])'''




data=pd.read_csv("/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/berkely_train.csv")
# data=pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/aptiveBB/reddy.csv')
print(data.head())
print(data.groupby('class').count())

# this is done to remove if xmin==xmax and ymin==yamax(which is actuallly wrong)
df=data[(data['xmin']!=data['xmax']) & (data['ymin']!=data['ymax'])]


print(df.head())

# df.to_csv("berkely_After_filter.csv")

# this is most important funtion to count no of class in group
new=data.groupby(['filename'])['class'].count()


gb = data.groupby(['filename'])
grouped_C = gb['class']
n=data.groupby(['filename', 'class'])
a=(n.size())
print(a)
gv=a.index[0]

for file_name, (cls) in enumerate(a):
    print(file_name)
    print(cls)
new1=data.groupby(['filename', 'class'])['xmin']#.count()

# b=data.groupby(level=['filname', 'class']).sum()




mydata=data.groupby('filename')
print(data.groupby('class').count())
len_group=mydata.ngroups
# index=mydata.groups['car'].values
mygroup=mydata.groups

# new=data.groupby(['filename', 'class'])#['car'].count()

# this is most important funtion to count no of class in group
new=data.groupby(['filename'])['class'].count()

for da in mygroup.keys():
    index = mydata.groups[da].values
    for read_index in index():


        print(index)
        print(da)
        break



'''for da in mydata.ngroups():

    index = mydata.groups['car'].values
    mydata.groups['0124dfa6-385f1b58'].values
    print(da)'''


# index=mydata.groups['car'].values
'''pyindex=np.random.choice(index, size=10000, replace=False)
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