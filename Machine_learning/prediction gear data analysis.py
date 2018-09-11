import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# import statsmodels.formula.api as smapi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
# base_path="Users\mayank\Documents\mytensorflow\Datasets\"

file_name="C:/Users/mayank/Documents/Datasets/MY_refine_engine_data.xlsx"
data=pd.read_excel(file_name)
'''
#ploting wheel rpm and gear
plt.scatter(data['WhlRPM_FL[rpm]']/10,data['Gr[]'],color='b')
# plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.xlabel("Wheel_RPM")
plt.ylabel("Gear")
plt.show()'''

corr_Data=(data.corr())
# same as this is heat map in seaborn

# sns.heatmap(corr_Data,annot=True)
sns.clustermap(corr_Data,annot=True)
# fig = sns.get_figure()
# fig.savefig("output.png")
#plt.savefig("myfig.png")
# plt.show()

# iris=sns.load_dataset('iris')#load inbuilt data sets like iris ,tips,flight
print (data.head())

sns.pairplot(data)
plt.show()
gr=sns.PairGrid(data)
gr.map_diag(sns.distplot)
gr.map_upper(plt.scatter)
gr.map_lower(sns.kdeplot)
plt.show()