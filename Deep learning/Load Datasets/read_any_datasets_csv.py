import pandas as pd
from batchup import data_source
import random
from sklearn.utils import shuffle

import numpy as np
df=pd.read_csv("/home/mayank-s/PycharmProjects/Datasets/pascal_voc_csv.csv")
print(df.head())
print (df.info())

df=df.iloc[np.random.permutation(len(df))]
print (df.head())
grouped = df.groupby('image_name')
bbox=[]
label=[]
image_name=[]
loop=0

# lis=[1,2,3,4,5,6]
# # random.shuffle(lis)
# for val in (random.shuffle(lis)):
#     print (val)
for name,group in (grouped):
    loop+=1
    print (name)
    image_name.append(name)
    # print (group)
    # print(group.iloc[0])
    # print (group.values)
    bbox_val=group.values[:,4:8]
    label_val= group.values[:,3]

    bbox.append(bbox_val)
    label.append(label_val)
    print(image_name)
    print (bbox)
    print(label)
    if loop>4000:
        break



print (bbox.__sizeof__())    #print(group.xmax.values)


sample_number= df.shape[0]
batch_size=1
batches = sample_number // batch_size
for i in range(batches):
    images = []
    boxes = []
    sample_index = np.random.choice(sample_number,batch_size,replace=True)
#ds = data_source.ArrayDataSource([image_name,bbox])
print(i)
'''for (im, bb) in ds.batch_iterator(batch_size=10, shuffle=True):
    print(im,bb)'''