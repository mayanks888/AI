import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
data=pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/Social_Network_Ads.csv')

print(data.head())

x = data.iloc[:, [2,3]].values
y = data.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 0)

#Here I have standardise the features (set the value between mean 0 and sd=1))
scale_val=StandardScaler()
scale_x=scale_val.fit_transform(X_train)
scale_c_test = scale_val.transform(X_test)
print("the scaler mean  is",scale_x[0].mean(),scale_x[1].mean())

# ###############################################333
# # ploting standard scaler in graph
# df = pd.DataFrame({
#                     'x1': x[0],
#                     'x2': x[1]
#                                 })
#
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
# scaled_df = pd.DataFrame(scale_x, columns=['x1', 'x2'])
# ax1.set_title('Before Scaling')
# sns.kdeplot(df['x1'], ax=ax1)
# sns.kdeplot(df['x2'], ax=ax1)
#
# ax2.set_title('After Standard Scaler')
# sns.kdeplot(scaled_df['x1'], ax=ax2)
# sns.kdeplot(scaled_df['x2'], ax=ax2)
#
# plt.show()
#
#
# ###############################################333

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(random_state=0)
log_reg.fit(scale_x,y_train)

y_pred=log_reg.predict(scale_c_test)  #predtion just take the index of argmax of maximum probaitly
print(y_pred)
new=     (log_reg.predict_proba(scale_c_test))
print(new[:,0])
print(log_reg.predict_proba(scale_c_test))#this jut show the probablity after logistic regression
print("training_score ",log_reg.score(X_train,y_train))
print("Testing score: " ,log_reg.score(X_test,y_test))

cm=confusion_matrix(y_test,y_pred)
print(cm)
#####################################################333
# visualising my classification data
'''from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train
X_set, y_set =X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, scale_val.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()    '''

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
#creating array of all the possible value between min and max of all feature in all dimension for ploting
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, scale_val.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())#this is for limit
# this is for ploting all value of feature and color them with red and green for visualisation
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()