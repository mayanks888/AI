import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
# from sklearn.modelselections import train_test_split
from sklearn.cross_validation import train_test_split

data=load_iris()
# print data


x_train,xtest,y_train,ytest=train_test_split(data.data,data.target,test_size=1,random_state=42)
print (x_train.shape)

DT=tree.DecisionTreeClassifier()
Mytree=DT.fit(x_train,y_train)

print ('train accuracy ', DT.score(x_train,y_train))
print ('test accuracy ', DT.score(xtest,ytest))

print (data.feature_names)
print (data.data)
print (data.target[1])
# import graphviz
# dot_data = tree.export_graphviz(Mytree, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")
#
# dot_data = tree.export_graphviz(Mytree, out_file=None,
#                          feature_names=data.feature_names,
#                          class_names=data.target_names,
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph = graphviz.Source(dot_data)
