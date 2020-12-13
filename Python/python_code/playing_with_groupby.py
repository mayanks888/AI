import pandas as pd

# data=pd.read_csv("/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/berkerly_new_useful/Berkerly_no_filter_include_all_classes.csv")
# data=pd.read_csv("/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/berkerly_new_useful/berkely_ready_to_train_for_retinanet_with_weather_pytorch.csv")
data = pd.read_csv(
    "/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/berkerly_new_useful/berkely_ready_to_train_for_retinanet_with_weather_pytorch.csv")
data = pd.read_csv("berkely_weather.csv")

df = data[(data['xmin'] != data['xmax']) & (data['ymin'] != data['ymax'])]

cool = (df.groupby('weather').count())
print(df.groupby('weather').count())

# classes=['person', 'traffic light', 'bus', 'car', 'motor','bike',  'traffic sign']
# classes=['weather','clear','foggy' ,'overcast', 'partly cloudy','rainy','snowy', 'undefined']
classes = ['weather', 'clear', 'overcast', 'partly cloudy', 'rainy', 'snowy']

# df=data.loc[data['class'].isin(["traffic sign","traffic light"])]
df = df.loc[data['weather'].isin(classes)]

print(df.groupby('weather').count())

df = df.replace("partly cloudy", "partly_cloudy")

# df=df.replace("bike", "cool")
# df=df.replace("cool", "motorbike")
# df=df.replace("traffic light", "traffic_light")
# df=df.replace("traffic sign", "traffic_sign")
print(df.groupby('weather').count())
# df.to_csv("berkely_weather.csv")
print(1)

# data=pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/aptiveBB/reddy.csv')
'''print(data.head())
print(data.groupby('class').count())
# this is done to remove if xmin==xmax and ymin==yamax(which is actuallly wrong)
# df=data[(data['xmin']!=data['xmax']) & (data['ymin']!=data['ymax'])]
# print(df.head())

grouped=data.groupby(['filename', 'class'])
a=(grouped.size())
filename=[]
car=[]
other=[]
for num,dat in enumerate(grouped.size()):
    print(dat)
    print(a.index[num][0])
    print(a.index[num][1])
    if a.index[num][1]=="car":
        filename.append(a.index[num][0])
        car.append(dat)

# df=pd.DataFrame()
df=pd.DataFrame(filename,car)#, columns=['filename','car'])
# df=df[filename,car]
# df.to_csv("Carcount.csv")'''

'''#now next job is to remove the selected index from this datasets based on car count on carcaount datasets
cardatsets=pd.read_csv('/home/mayank-s/PycharmProjects/Data_Science_New/Python/python_code/maxcarcaount_3.csv')
print(cardatsets.head())
filename=cardatsets.iloc[:,1].values

ran_file=np.random.choice(filename, size=2000, replace=False)
print(1)

flag=0
loop = 0
new=data.groupby('class')
print(data.groupby('class').count())
while(1):

    # print(ran_file[loop])
    # print(data.groupby('class').count())
    # data.drop(data[ran_file[0]], inplace=True)
    # data = data.drop(ran_file[loop], axis=0)

    # df[df.name != 'Tina']
    df=data=data[data.filename!=ran_file[loop]]
    count_val=data.groupby('class').count()
    # print(count_val.iloc[2, 1])
    if count_val.iloc[2,1] < 35000:
        print(data.groupby('class').count())
        data.to_csv("Berkerly_final_Dataseets.csv")
        break
    loop=loop+1

# pyindex=np.random.choice(index, size=30000, replace=False)
# data.drop(data.index[pyindex],inplace=True)
# print(data.groupby('class').count())'''
