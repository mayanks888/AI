import pandas as pd
import numpy as np

data=pd.read_csv('train.csv')
# removing column and entire data rows from datasets
newdata = data.drop('Name',axis=1)
newdata = data.drop(labels=['Name',"Age"],axis=1)
print newdata.head()
print data.head()

# dropping a particular columns based on a column index
print data.drop(labels=data.columns[[0,4]],axis=1).head()

#droping the duplicate values

print data['Embarked'].head()

# printing data not related to "s"
print data[data['Embarked']!='S'].head()
print data['Embarked'].unique()


#i could not understand delete duplicate function
data_embarked=data['Embarked']
print data_embarked.drop_duplicates(subset=['Embarked']).head()

groupby in dataframe
print data.groupby("Age").mean()#so we will get the group of only numerical data
print data.groupby(["Age","Sex"])['Survived','Pclass'].mean()#groupby only specific column

print data['Name'].upper()#not working

for name_data in data['Name']:
    print name_data.upper( )#upper class name

print data.drop(labels=data.columns[[0,4]],axis=1).head()