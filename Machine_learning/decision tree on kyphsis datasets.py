import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
data=pd.read_csv("C:/Users/mayank/Documents/Datasets/kyphosis.csv")
print (data.head())

x=data.drop('Kyphosis',axis=1)
y=data['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 0)

mydtree=DecisionTreeClassifier()
mydtree.fit(X_train,y_train)
predict_val=mydtree.predict(X_test)
print (classification_report(y_test,predict_val))
print (confusion_matrix(y_test,predict_val))



from sklearn import tree
import pydot
import collections
# data_feature_names = [ 'height', 'hair length', 'voice pitch' ]
# dot_data = tree.export_graphviz(mydtree,
#                                 feature_names=None,
#                                 out_file=None,
#                                 filled=True,
#                                 rounded=True)
# graph = pydot.graph_from_dot_data(dot_data)
#
# colors = ('turquoise', 'orange')
# edges = collections.defaultdict(list)

from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree
import pydot

features=list(data.columns[1:])

dot_data=export_graphviz(mydtree, out_file=None,
                feature_names=features,
                filled=True, rounded=True,
                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('dtree_render',view=True)


#
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
# import pydotplus as pydot
import matplotlib.image as mpimg

features=list(data.columns[1:])
dot_data = StringIO()
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files(x86)/Graphviz2.38/bin'

export_graphviz(mydtree, out_file=dot_data,
                feature_names=features,
                filled=True, rounded=True,
                special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())