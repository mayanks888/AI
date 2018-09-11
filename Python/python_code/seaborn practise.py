import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('C:/Users/mayank/Documents/Datasets/Salaries.csv')
data2=pd.read_csv("C:/Users/mayank/Documents/Datasets/Startups.csv")
dat=data.head(100)
print(data2.head())
# distribution plot
my_data=sns.distplot(data['TotalPay'],kde=False,bins=40)
plt.show()

sns.jointplot(x='Year',y='TotalPay',data=dat)
plt.show()

#pair plot important
sns.pairplot(dat['TotalPay'])
plt.show()

sns.barplot(y='R&D Spend',x='State',data=data2)#bar plot show mean of data used
plt.show()

sns.countplot(x='State',data=data2)#plot the count of the numver
plt.show()

#box plot or whisker plot
sns.boxplot(x='State',y='R&D Spend',data=data2)
plt.show()

#voilen plot
sns.violinplot(x='State',y='R&D Spend',data=data2,split=True)
plt.show()

#strip plot
sns.stripplot(x='State',y='R&D Spend',data=data2,jitter=True)#,hue='Florida',split=True)
plt.show()

#print any plot
sns.factorplot(x='State',y='R&D Spend',data=data2,kind='bar')
plt.show()

# correlance data
corr_Data=(data2.corr())
# same as this is heat map in seaborn

# sns.heatmap(corr_Data,annot=True)
sns.clustermap(corr_Data,annot=True)
plt.show()


matrixform=data2.pivot_table(index='State',columns='Administration',values='Profit')
print(matrixform)
sns.heatmap(matrixform,annot=True)
plt.show()

#regression plot
# sns.lmplot(x='Profit',y='R&D Spend',data=data2,hue='State')
sns.lmplot(x='Profit',y='R&D Spend',data=data2)

plt.show()

iris=sns.load_dataset('iris')#load inbuilt data sets like iris ,tips,flight
print (iris.head())

sns.pairplot(iris)
plt.show()
gr=sns.PairGrid(iris)
gr.map_diag(sns.distplot)
gr.map_upper(plt.scatter)
gr.map_lower(sns.kdeplot)
plt.show()

#working on tips datasets
tips=sns.load_dataset('tips')
print (tips.head())

gpr=sns.FacetGrid(data=tips,col='time',row='smoker')
gpr.map(sns.distplot,'total_bill')
plt.show()




