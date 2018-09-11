import pandas as pd
import numpy as np

myseries =pd.Series([98,7,np.NAN,8,"hello",2,"mayank",np.NAN,5])
print myseries

#check if series is empty


print myseries.isnull()
print  "the null count is",myseries.isnull().sum()
print myseries.notnull()
print  "the not null count is",myseries.notnull().sum()

#lets use our data sets
data=pd.read_csv('train.csv')
print data['Age'].isnull().sum()
print data['Age'].notnull().sum()
print data['Survived'].isnull().sum()
print data['Survived'].notnull().sum()
print data['Survived'].count()
#lets replace NAN with some value eg pizza
new_data=data["Age"].replace(np.nan,'pizza')
print new_data.head()#here we have removed all the nan  with male
print new_data.isnull().sum()
print new_data


