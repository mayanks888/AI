#groupby by timestand
import pandas as pd
import numpy as np

ts_data=pd.date_range('01/01/2018',periods=200000,freq='60S')
# print ts_data

df=pd.DataFrame()
df['timestand']=ts_data
# print df.head()
new_df=df.set_index(df['timestand'])


new_df=pd.DataFrame(index=ts_data)
print new_df.head()

find_random=np.random.randint(1,10,200000)
new_df["vehicle count"]= find_random

print new_df.head(10)


# we will sample the data with respects to week wise
# T means = minute
# W=Weeks
# M=Monthly
# so we are sample total number of cars in a day
print new_df.resample('1440T').sum()
print new_df.resample('M').sum()

