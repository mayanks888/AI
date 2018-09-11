import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
data=pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/Social_Network_Ads.csv')
print(data.head())

x = data.iloc[:, [2,3]].values
y = data.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .25, random_state = 0)

#Here I have standardise the features (set the value between mean 0 and sd=1))
scale_val=StandardScaler()
scale_x=scale_val.fit_transform(X_train)
scale_c_test = scale_val.transform(X_test)
print("the scaler mean  is",scale_x[0].mean(),scale_x[1].mean())

k_classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)#p=2 maean we will use eucleadean distance

k_classifier.fit(scale_x,y_train)

y_pred=k_classifier.predict(scale_c_test)
print(y_pred)

print("training_score ",k_classifier.score(X_train,y_train))
print("Testing score: " ,k_classifier.score(X_test,y_test))

cm=confusion_matrix(y_test,y_pred)
print(cm)