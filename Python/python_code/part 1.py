import pandas as pd
import numpy as np

#hey there if repo thing is working
#now github to pycharm
# ----------------------------------------------------------------------
                        # DataFrame
#how to create a data frame
df=pd.DataFrame()
df['name']=['mayank','shashank']
df['place']=['bangalore','delhi']
df['profession']=['engineer','HR']
print df
print type(df)

#now if you want to add new rows into dataframe##
new_row=pd.Series(['sandy','bangalore','computer engineer'])
#adding rows to dataframe
print df.append(new_row,ignore_index=True)

# if you will observe that index is not proper so now you need give index to your data series
new_row=pd.Series(['sandy','bangalore','computer engineer'],index=['name','place','profession'])
# new_row=pd.Series(['sandy','bangalore','computer engineer'],index=[0,1,2])#not working this way
print df.append(new_row,ignore_index=True)

# changing index of dataframe and assigning it to new dataframe
cf= df.set_index(df['name'])

#now you can use loc command to select rows of dataframe of that name
print cf.loc['mayank']
# ----------------------------------------------------------------------
                        # Datasets

datasets=t=pd.read_csv('train.csv')
print datasets.head()

#how to describe datasets
print datasets.shape
# to read particular row in datasets
print datasets.iloc[5]
#to read pardicular data at 5 rows and 3 cdlumn
print datasets.iloc[5][3]

#finding particular columns from datasets
print datasets[['PassengerId','Name','Age']].head()

#finding rows based on some condition
print datasets[datasets['Age']>25].head()
print datasets[(datasets['Pclass']>1) & (datasets['Age']>25)].head()

#how to replace a value in datasets
#here we addind same columnn with updated datasets
datasets['Sex']=datasets['Sex'].replace('male','men').head()
print datasets.head()
# print datasets['Sex'].replace('male','men').head()
#adding it in our datasets with new column
datasets['new column']=datasets['Sex'].replace('male','men')
print datasets.head()

#changing column name in datasets
# df.rename(index=str, columns={"A": "a", "B": "c"})
print datasets.rename(columns={'new column':'my column'}).head()

datasets['Sex']=datasets['Sex'].replace('male','men').head()
print datasets.head()

# find max value of datasets
print 'the max age= ', datasets['Age'].max()
print 'the minimum age= ',datasets['Age'].min()
print 'the sum of all ags=',datasets['Age'].sum()
# finding the number of rows in particular age and this will also contain "NAN' data included
print 'total rows in age', datasets["Age"].count()
print 'the average(mean) is ',datasets["Age"].mean()
print 'the standard deviation is ',datasets["Age"].std()
print ' the unique value in datasets colume Sex', datasets["Sex"].unique()
#the value count function with give a count of each unique value
print ' the unique value in datasets colume Sex', datasets["Sex"].value_counts()
#

