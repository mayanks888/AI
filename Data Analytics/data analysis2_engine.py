import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smapi
from sklearn.ensemble import RandomForestClassifier

data=pd.read_excel("MY_refine_engine_data.xlsx")
'''
#ploting wheel rpm and gear
plt.scatter(data['WhlRPM_FL[rpm]'],data['Gr[]'],color='b')
# plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.xlabel("Wheel_RPM")
plt.ylabel("Gear")
plt.show()

#ploting wheel rpm and gear
plt.scatter(data['EngRPM[rpm]'],data['Gr[]'],color='r')
# plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.xlabel("Engine_RPM")
plt.ylabel("Gear")
plt.show()
#ploting wheel accped and gear
plt.scatter(data['AccelPdlPosn[%]'],data['Gr[]'],color='b')
# plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.xlabel("Accped")
plt.ylabel("Gear")
plt.show()
#ploting wheel engine torque and gear
plt.scatter(data['EngTrq[Nm]'],data['Gr[]'],color='g')
# plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.xlabel("Engine Torque")
plt.ylabel("Gear")
plt.show()'''

#ploting wheel engine torque and gear
plt.scatter(data['WhlRPM_FL[rpm]'],data['AccelPdlPosn[%]'],color='g')
# plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.xlabel("Engine Torque")
plt.ylabel("Gear")
plt.show()


plt.scatter(data['EngRPM[rpm]'],data['EngTrq[Nm]'],color='g')
# plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.xlabel("EngRPM")
plt.ylabel("EngTrq")
plt.show()

# features=data.iloc[:,:-1].values
# labels=data.iloc[:,-1].values
# print features
# # print labels
