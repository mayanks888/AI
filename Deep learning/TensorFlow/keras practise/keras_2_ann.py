from keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
#dataset=pd.read_csv('../../../../Datasets/Churn_Modelling.csv')
dataset=pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/Churn_Modelling.csv')

print (dataset.head())
# dataset['Geography']= (dataset['Geography'].str.strip())
# dataset['Gender']= (dataset['Gender'].str.strip())
# print (dataset['Geography'].unique)
# print (dataset['Gender'].unique)
# print (pd.get_dummies(dataset['Gender']))
# print (pd.get_dummies(dataset['Geography']))
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
X[:, 1] = labelencoder_x_1.fit_transform(X[:, 1])#X[:,1] means all rows and first column
labelencoder_x_2 = LabelEncoder()
X[:, 2] = labelencoder_x_2.fit_transform(X[:, 2])
print (X[:,1])
print (X[:,2])

oneHotencoder = OneHotEncoder(categorical_features = [1])
X = oneHotencoder.fit_transform(X).toarray()

X=X[:,1:]

# -----------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
doscaling= StandardScaler()
doscaling.fit(X)#this was done to remove target class from data frame
# # StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_feature=doscaling.transform(X)
X=scaled_feature #using same data as after scaling
# print scaled_feature

# -------------------------------------------------------------------------
cb=TensorBoard(log_dir=("/home/mayank-s/PycharmProjects/Data_Science/output_graph/try3"))#,histogram_freq = 1, batch_size = 32,write_graph ="TRUE" )

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .20, random_state = 0)
print (X_train.shape)
from keras.models import Sequential
from keras.layers import Dense
myclaisfer=Sequential()

myclaisfer.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
myclaisfer.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
myclaisfer.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
myclaisfer.summary()
out = myclaisfer.layers[(0)].output#reading any dense layer
#weight_out = myclaisfer.get_layer(0).get_weights()[0]#this will showthe weight asssign to dense layer 
myclaisfer.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# myclaisfer.fit(X_train,y_train,batch_size=10,nb_epoch=3,  callbacks = callback_tensorboard("/home/mayank-s/PycharmProjects/output_graph"))
myclaisfer.fit(X_train,y_train,batch_size=10,nb_epoch=3,  callbacks=[cb])
y_predict=myclaisfer.predict(X_test)
print(y_predict)
y_predict=(y_predict>0.50)
print (confusion_matrix(y_test,y_predict))

print (out.value)