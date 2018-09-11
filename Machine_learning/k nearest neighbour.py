import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
data = pd.read_csv("Classified Data.xls")
print (data.head())
doscaling= StandardScaler()
doscaling.fit((data.drop('TARGET CLASS',axis =1)))#this was done to remove target class from data frame
# # StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_feature=doscaling.transform(data.drop('TARGET CLASS',axis =1))
# print scaled_feature
new_data= pd.DataFrame(scaled_feature,columns=data.columns[:-1])#this was done to maintain column name without target class
# print new_data.head()

X=new_data
y=data["TARGET CLASS"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 0)

# mykclass=KNeighborsClassifier(n_neighbors=5)
# mykclass.fit(X_train,y_train)
# predict_val=mykclass.predict(X_test)
#  print classification_report(y_test,predict_val)
# print confusion_matrix(y_test,predict_val)

error_class=[]#lets find error in total algoristhm
for lop in range(1,40):
    mykclass = KNeighborsClassifier(n_neighbors=lop)
    mykclass.fit(X_train, y_train)
    predict_val=mykclass.predict(X_test)
    error_class.append(np.mean(predict_val!=y_test))
print (error_class)

#lets plot rhe error class
plt.plot(range(1,40),error_class,color='b',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.xlabel('k value')
plt.ylabel('error rate')
plt.show()




