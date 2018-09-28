import pandas as pd
from collections import namedtuple, OrderedDict
data=pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/berkely_train.csv')

print(data.head())

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

grouped = split(data, 'filename')
new_data= (data.groupby('class'))
# cool_Data=new_data["traffic sign","traffic light"]
# print (new_data.head())
for group in grouped:
    # print(group[0])
    if (group[0]=="traffic sign" or group[0]=="traffic light"):
        group_data=group.object
        # print('2')
# print(1)
#2nd step
# df=data[data['class']==["traffic sign","traffic light"]]# or (data['class']=="traffic light")]
# df=data[(data['class']=="traffic sign") or (data['class']=="traffic light")]
# print[data.head]
# datasets[(datasets['Pclass']>1) & (datasets['Age']>25)].head()
# df=data.loc[data['class'] == ("traffic sign","traffic light")]
df=pd.DataFrame
# df=data.loc[data['class'].isin(["traffic sign","traffic light"])]
df=data.loc[data['class'].isin(["car"])]
# (df.loc[df['B'].isin(['one','three'])]


threshold=850

cool=df['xmax']-df['xmin']> threshold

widht=df[df['xmax']-df['xmin'] > threshold]
height=df[df['ymax']-df['ymin'] > threshold]
# sort=df[(df['xmax']-df['xmin'] > threshold) or (df['ymax']-df['ymin'] > threshold)]
# print(1)
total=0
lim=df.shape[0]
for dat in range(lim):
    # print(dat)
    height=df.iloc[dat,7]-df.iloc[dat,5]
    widht = df.iloc[dat,6] - df.iloc[dat,4]
    if (height>threshold or widht> threshold):
        print("image_name : {im} , class : {cl}\t, height_dff : {hd} , width diff : {wd}".format(im=df.iloc[dat,0],cl=df.iloc[dat,3],hd=height,wd=widht))
        total+=1

print("\ntotal defect identified",total)