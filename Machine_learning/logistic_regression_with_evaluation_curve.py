# Logistic Regression

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# this is for showing curve accuracy vs with different no of sample
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(penalty='l2', random_state=0))])
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.3, 1.0])
plt.show()
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
# regularisation parameter vs accuracy roc_curve()
from sklearn.learning_curve import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    param_name='clf__C',
    param_range=param_range,
    cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5,
         label='training accuracy')
plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')
plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')
plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.show()

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print('the confusion matrix is ', cm)
# we will go deep inside evaluation

# Precison=(TP/(TP+FP))
# Recall(true positive rate)=(TP/(TP+FN))
from sklearn.metrics import precision_score, recall_score, f1_score

print('Precisoion is ', precision_score(y_test, y_pred))
print('Recall (TPR) is', recall_score(y_test, y_pred))
print('F1 score is ', f1_score(y_test, y_pred))

# precisiion-recal vs threshold curve
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([-0.5, 1.5])


plt.figure(figsize=(12, 8));
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# precisiion vs recal at diff threshold curve
plt.figure(figsize=(12, 8));
plt.plot(precisions, recalls);
plt.xlabel('recalls');
plt.ylabel('precisions');
plt.title('PR Curve: precisions/recalls tradeoff');
plt.show()

# checking roc curve now
from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve(y_test, y_pred)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')


plt.figure(figsize=(12, 8));
plot_roc_curve(fpr, tpr)
plt.show();

# area under the curve
from sklearn.metrics import roc_auc_score

print("the area under the curve", roc_auc_score(y_test, y_pred))

# Use PR curve whenever the **positive class is rare** or when you care more about the false positives than the false negatives
#
# Use ROC curve whenever the **negative class is rare** or when you care more about the false negatives than the false positives

# In the example above, the ROC curve seemed to suggest that the classifier is good. However, when you look at the PR curve, you can see that there are room for improvement.


# last is combined all score into one known as classification score
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# Visualising the Training set results
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
a = 1
X1, X2 = np.meshgrid(np.arange(start=a * ((X_set[:, 0].min() - 1)), stop=a * (X_set[:, 0].max() + 1), step=0.01),
                     np.arange(start=a * X_set[:, 1].min() - 1, stop=a * X_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))


new = (np.array([X1.ravel(), X2.ravel()]).T)

cool3 = (classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape))
# Trying to print probabily into the grapg instead of just 1 and 0
prob_val = (classifier.predict_proba(np.array([X1.ravel(), X2.ravel()]).T)[:, 1].reshape(X1.shape))
# prob_val=(classifier.predict_proba(np.array([X1.ravel(), X2.ravel()]).T))[:,0]
plt.contourf(X1, prob_val, X2,
             alpha=0.50, cmap=ListedColormap(('red', 'green')))
# plt.contourf(X1, X2, classifier.predict_proba(np.array([X1.ravel(), X2.ravel()]).T)[0].reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# log_reg.predict_proba(scale_c_test)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
# plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#             c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
