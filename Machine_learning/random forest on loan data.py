import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
data=pd.read_csv('loan_data.csv')
print data.head()

x=data.drop('not.fully.paid',axis=1)
y=data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 0)

mydtree=DecisionTreeClassifier()
mydtree.fit(X_train,y_train)
predict_val=mydtree.predict(X_test)
print classification_report(y_test,predict_val)
print confusion_matrix(y_test,predict_val)

