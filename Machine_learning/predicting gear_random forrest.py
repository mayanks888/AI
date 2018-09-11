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
from sklearn.tree import DecisionTreeClassifier
# base_path="Users\mayank\Documents\mytensorflow\Datasets\"

file_name="C:/Users/mayank/Documents/Datasets/MY_refine_engine_data.xlsx"
data=pd.read_excel(file_name)
'''
#ploting wheel rpm and gear
plt.scatter(data['WhlRPM_FL[rpm]']/10,data['Gr[]'],color='b')
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

features=data.iloc[:,:-1].values
labels=data.iloc[:,-1].values
print (features)
# print labels

x_train,xtest,y_train,ytest=train_test_split(features,labels,test_size=.25,random_state=2355)
print (x_train.shape)
# mydtree=DecisionTreeClassifier(random_state=2355)
# mydtree.fit(x_train,y_train)
myclass=RandomForestClassifier(n_estimators=100, oob_score=True,random_state=2355)
myclass.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
predicted = myclass.predict(xtest)
print ("predicted Gear :", predicted)
print
print("Real Gear are", ytest)
accuracy = accuracy_score(ytest, predicted)
print
print ("the Accuracy of Model is : ",accuracy)

# #ploting wheel engine torque and gear
# plt.scatter(ytest,predicted,color='g')
# # plt.plot(x_train,regresion_data.predict(x_train),color='r')
# plt.xlabel("Predicted Gear")
# plt.ylabel("Real Gear")
# plt.show()


# print (myclass.max_leaf_nodes)
df=pd.DataFrame()
df['Real_gear']= ytest
df['predicted gear']= predicted


# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(ytest)
# z = le.fit_transform(predicted)
dat123=['0','1','2','3','4','5','6','7','8','13']
dat12=[0,1,2,3,4,5,6,7,8,13]
labedled = [dat12,dat12]
cm=( confusion_matrix(ytest, predicted, labels=dat12))
print ("Confuction Matrix:\n",cm)
# print (confusion_matrix(y, z, labels=labedled))
# "________________________________________________"

'''import seaborn as sns

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted Gear');ax.set_ylabel('True Gear')
ax.set_title('Confusion Matrix')

ax.xaxis.set_ticklabels(dat123); ax.yaxis.set_ticklabels(dat123)
plt.show()'''
# ___________________________________________________
# Compute confusion matrix
# cm = confusion_matrix(ytest, predicted)
# np.set_printoptions(precision=2)
# print('Confusion matrix, without normalization')
# print(cm)
# plt.figure()
# plot_confusion_matrix(cm)
# writer = pd.ExcelWriter('finalprediction1.xlsx', engine='xlsxwriter')
#
# # Convert the dataframe to an XlsxWriter Excel object.
# df.to_excel(writer, sheet_name='Sheet1')
#
# # Close the Pandas Excel writer and output the Excel file.
# writer.save()

# __________________________________________________________
'''from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree
import pydot

features=list(data.columns[1:])

dot_data=export_graphviz(mydtree, out_file=None,
                feature_names=features,
                filled=True, rounded=True,
                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('dtree_render',view=True)'''
# __________________________________________________________

#
'''from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
# import pydotplus as pydot
import matplotlib.image as mpimg

features=list(data.columns[1:])
dot_data = StringIO()
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files(x86)/Graphviz2.38/bin'

export_graphviz(myclass, out_file=dot_data,
                feature_names=features,
                filled=True, rounded=True,
                special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())'''
# __________________________________________________________________