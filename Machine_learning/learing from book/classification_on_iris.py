import pandas as pd
import numpy as np
from  sklearn import  datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,precision_score,recall_score
mydataset=datasets.load_iris()

features=mydataset.data
labels=mydataset.target
print(labels)

scaling=StandardScaler()
features=scaling.fit_transform(features)
X=features[:,[1,2]]
y=labels
print('classes:', np.unique(labels))#types of classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#visualising x train and y train
import matplotlib.pyplot as plt
#ploting wheel rpm and gear
plt.scatter(X_train[:,0],X_train[:,1],color='r')
# plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.xlabel("sepal length")
plt.ylabel("petal length")
# plt.show()

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=1, eta0=0.1, random_state=0)
ppn.fit(X_train, y_train)

y_pred=(ppn.predict(X_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print(confusion_matrix(y_test,y_pred))
print('the presion is ', precision_score(y_test,y_pred,average='macro'))
print("recall is ",recall_score(y_test,y_pred,average='macro'))
#last is combined all score into one known as classification score
print(ppn.predict_proba(X_test))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier,test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),
        marker=markers[idx], label=cl)
        # highlight test samples
        if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1], c='',
            alpha=1.0, linewidths=1, marker='o',
            s=55, label='test set')

X_combined_std = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
y=y_combined,
classifier=ppn,
test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()