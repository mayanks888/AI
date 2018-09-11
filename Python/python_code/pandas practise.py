import pandas as pd
import numpy as np
'''#basically excel version of python
mylist=[1,2,3,4]
ind=['a','b','c','d']
data=pd.Series(mylist,index=ind)#similar like excel as index is serial number
print (data)
print (data['a'])
data.columns='mynum'
print (data.columns)'''

#now lets learn about data frame
# daraframe is nothing  but a collection of series just like excel(bunch of column)
val=np.random.seed(10)
print (val)
row=['a','b','c','d']
data3='abcd'
row=data3.split()#fisrt method of split
row=list(data3)#second method of splits
colm=['e','f','g','h','i']
data2=np.random.randn(4,5)#4 rows 5 column

df=pd.DataFrame(data2,index=row,columns=colm)#define dataframe like excel
'''# df['colm']=row
# print(data2)
print (df)
# print (df['e']['a'])#this will show data at 'e' colum and 'a' row
# print(df.iloc[1:,2:4])
# print(df.drop(labels='e[]',axis=1))#delete the rows or column from dataframe based on a dataframe and inplace =true if wany permenant
# print (df[df>0])#very important
# print (df[df['g']>0])#so it will only  show rows g whose column does satisfy the condition
# print (df[['e','f']])#when you want to uses two column in dataframe use two square brackets
# print (df[(df['g']>0) & (df['h']>1)])
# print (df[(df['g']>0) | (df['h']>1)])#you have to use '|0and & ' instesd od 'and' and or
#'''

#missing value

'''df['j']=[1,2,3,np.nan]
print (df)
print (df.dropna(axis=0))#droping nan value based on index i.e axis=0 if rows and 1 for column
print(df.fillna('pizza'))
print(df.fillna(df['j'].mean()))#fill nan with mean of column j'''


#groupby
#do it youself

#merge and concanate
row=['a','b','c','d']
data3='abcd'
row=data3.split()#fisrt method of split
row=list(data3)#second method of splits
colm=['e','f','g','h','i']
data3=np.random.randn(4,5)#4 rows 5 column

df2=pd.DataFrame(data3,index=row,columns=colm)
# print (df,"\n",df2)
# print(pd.concat([df,df2],axis=1))
print (df)
#operations

# print (df.nunique())#find the unique valuein per column

'''print (df['e'].value_counts())#count no of values repeated in colums 'e'
print (df['e'].sum())#sum of column

#what if u want to apply your function in dataframe

print (df.apply(lambda val:val**2))#define you own function based on a lambda and use insode dataframe
'''
print (df['e'].sort_values())#sort only column e
print (df.sort_values(by ='e'))#sort all frame based on column e

#learn pivot table]