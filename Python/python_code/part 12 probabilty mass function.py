import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt

total_dat_=5000
data=np.random.randint(0,100,total_dat_)
# example=[2 7 5 6 5 8 2 1 4 5 3 2 3 0 9 8 4 9 6 6 6 9 1 0 5 7 4 1 6 4 4 0 0 2 0 7 4 9 0 1 3 5 7 5 8 7 2 3 4 8]
print data
df=pd.DataFrame(data)
print df[0].value_counts()

df2=pd.DataFrame()
df2['random number']=df[0].value_counts()
total=df[0].sum()
print total

df2["probability"]=df2['random number']/total_dat_
print df2
print df2["probability"].sum()

# plt.bar(df2['random number'],df2["probability"])
plt.hist(df2['random number'],df2["probability"],bins=20)
plt.show()


