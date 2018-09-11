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

features=data.iloc[:,:-1].values
labels=data.iloc[:,-1].values
print (features)
# print labels

x_train,xtest,y_train,ytest=train_test_split(features,labels,test_size=.25,random_state=2355)
print (x_train.shape)
mydtree=DecisionTreeClassifier(random_state=2355)
mydtree.fit(x_train,y_train)
myclass=RandomForestClassifier(random_state=2355)#n_estimators=100, oob_score=True,random_state=2355)
myclass.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
predicted = mydtree.predict(xtest)
print ("predicted Gear :", predicted)
print
print("Real Gear are", ytest)
accuracy = accuracy_score(ytest, predicted)
print
print ("the Accuracy of Model is : ",accuracy)
# __________________________________________________________
# here you seee decison tree
from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree
import pydot
# features=list[data.iloc[:,:-1].values]
features=list(data.columns[0:4])
# features=list(data.columns[1:])
dat123=['0','1','2','3','4','5','6','7','8','13']
dot_data=export_graphviz(mydtree,class_names=dat123, out_file=None,
                feature_names=features,
                filled=True, rounded=True,
                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('dtree_render',view=True)
# __________________________________________________________
