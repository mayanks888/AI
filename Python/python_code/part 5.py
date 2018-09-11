import pandas as pd
import numpy as np

df1=pd.DataFrame()
df2=pd.DataFrame()

name=['Mayank','sandy','vivek']
place=['bangalore', 'delhi','dehradun']
id=[101,102,105]

# df1['Id','name','place']=[id,name,place]
df1=pd.DataFrame({'ID':id,'name':name,'place':place})
print df1

id=[102,105,106]
profession=['engineer','HR','SE']
df2=pd.DataFrame({'ID':id,'profession':profession})
print df2

df3=pd.DataFrame()
'''how : {‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘inner’

 left: use only keys from left frame, similar to a SQL left outer join; preserve key order
 right: use only keys from right frame, similar to a SQL right outer join; preserve key order
outer: use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically
 inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys
this means we are merging on "ID"'''


df3=pd.merge(df1,df2,on="ID")
print df3

df3=pd.merge(df1,df2,on="ID",how='outer')
print df3

df3=pd.merge(df1,df2,on="ID",how='left')
print df3

df3=pd.merge(df1,df2,on="ID",how='right')
print df3