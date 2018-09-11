import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data1=pd.read_csv('df1')
print (data1.head())

# pandas builin visualisations/ basically it is matplotlib builtlib function
# data1['A'].plot.hist(bins=50)
data1.plot.scatter('A','B')
plt.show()
data1.iplot