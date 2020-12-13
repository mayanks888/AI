# coding: utf-8

# # Classification Based Machine Learning Algorithm
# 
# [An introduction to machine learning with scikit-learn](http://scikit-learn.org/stable/tutorial/basic/tutorial.html#introduction)

# ## Scikit-learn Definition:
# 
# **Supervised learning**, in which the data comes with additional attributes that we want to predict. This problem can be either:
# 
# * **Classification**: samples belong to two or more *classes* and we want to learn from already labeled data how to predict the class of unlabeled data. An example of classification problem would be the handwritten digit recognition example, in which the aim is to assign each input vector to one of a finite number of discrete categories. Another way to think of classification is as a discrete (as opposed to continuous) form of supervised learning where one has a limited number of categories and for each of the n samples provided, one is to try to label them with the correct category or class.
# 
# 
# * **Regression**: if the desired output consists of one or more *continuous variables*, then the task is called regression. An example of a regression problem would be the prediction of the length of a salmon as a function of its age and weight.

# MNIST dataset - a set of 70,000 small images of digits handwritten. You can read more via [The MNIST Database](http://yann.lecun.com/exdb/mnist/)

# ***

# ## Downloading the MNIST dataset

# In[1]:


import numpy as np

# In[2]:


from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')

# In[3]:


mnist

# In[4]:


len(mnist['data'])

# # Visualisation

# In[5]:


X, y = mnist['data'], mnist['target']

# In[6]:


X

# In[7]:


y

# In[8]:


X[69999]

# In[9]:


y[69999]

# In[10]:


X.shape

# In[11]:


y.shape

# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# In[13]:


_ = X[1000]
_image = _.reshape(28, 28)
plt.imshow(_image);

# In[14]:


y[1000]

# ### Exercise: Locating the number 4 and plot the image

# In[15]:


type(y)

# In[16]:


y == 4

# In[17]:


np.where(y == 4)

# In[18]:


y[24754]

# In[19]:


_ = X[24754]
_image = _.reshape(28, 28)
plt.imshow(_image);

# ***

# # Splitting the train and test sets

# In[20]:


num_split = 60000

X_train, X_test, y_train, y_test = X[:num_split], X[num_split:], y[:num_split], y[num_split:]

# **Tips**: Typically we shuffle the training set. This ensures the training set is randomised and your data distribution is consistent. However, shuffling is a bad idea for time series data.

# # Shuffling the dataset

# [Alternative Method](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html)

# In[21]:


import numpy as np

# In[22]:


shuffle_index = np.random.permutation(num_split)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# ## Training a Binary Classifier

# To simplify our problem, we will make this an exercise of "zero" or "non-zero", making it a two-class problem.
# 
# We need to first convert our target to 0 or non zero.

# In[23]:


y_train_0 = (y_train == 0)

# In[24]:


y_train_0

# In[25]:


y_test_0 = (y_test == 0)

# In[26]:


y_test_0

# At this point we can pick any classifier and train it. This is the iterative part of choosing and testing all the classifiers and tuning the hyper parameters

# ***

# # SGDClassifier
# 
# # Training

# In[27]:


from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(random_state=0)
clf.fit(X_train, y_train_0)

# # Prediction

# In[28]:


clf.predict(X[1000].reshape(1, -1))

# ***

# # Performance Measures
# 
# # Measuring Accuracy Using Cross-Validation
# 
# ## StratifiedKFold

# Let's try with the `StratifiedKFold` stratified sampling to create multiple folds. At each iteration, the classifier was cloned and trained using the training folds and makes predictions on the test fold. 

# StratifiedKFold utilised the Stratified sampling concept
# 
# * The population is divided into homogeneous subgroups called strata
# * The right number of instances is sampled from each stratum 
# * To guarantee that the test set is representative of the population

# In[29]:


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

clf = SGDClassifier(random_state=0)

# In[30]:


skfolds = StratifiedKFold(n_splits=3, random_state=100)

# In[31]:


for train_index, test_index in skfolds.split(X_train, y_train_0):
    clone_clf = clone(clf)
    X_train_fold = X_train[train_index]
    y_train_folds = (y_train_0[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_0[test_index])

    clone_clf.fit(X_train_fold, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print("{0:.4f}".format(n_correct / len(y_pred)))

# #### `cross_val_score` using K-fold Cross-Validation

# K-fold cross-validation splits the training set into K-folds and then make predictions and evaluate them on each fold using a model trained on the remaning folds.

# In[32]:


from sklearn.model_selection import cross_val_score

# In[33]:


cross_val_score(clf, X_train, y_train_0, cv=3, scoring='accuracy')

# #### Exercise:
# 
# What if you would like to perform 10-fold CV test? How would you do that

# In[34]:


cross_val_score(clf, X_train, y_train_0, cv=10, scoring='accuracy')

# ***

# ## Danger of Blindly Applying Evaluator As a Performance Measure

# Let's check against a dumb classifier

# In[35]:


1 - sum(y_train_0) / len(y_train_0)

# A simple check shows that 90.1% of the images are not zero. Any time you guess the image is not zero, you will be right 90.13% of the time.
# 
# Bare this in mind when you are dealing with **skewed datasets**. Because of this, accuracy is generally not the preferred performance measure for classifiers.

# # Confusion Matrix

# In[36]:


from sklearn.model_selection import cross_val_predict

# In[37]:


y_train_pred = cross_val_predict(clf, X_train, y_train_0, cv=3)

# In[38]:


from sklearn.metrics import confusion_matrix

# In[39]:


confusion_matrix(y_train_0, y_train_pred)

# Each row: actual class
# 
# Each column: predicted class
# 
# First row: Non-zero images, the negative class:
# * 53360 were correctly classified as non-zeros. **True negatives**. 
# * Remaining 717 were wrongly classified as 0s. **False positive**
# 
# 
# Second row: The images of zeros, the positive class:
# * 395 were incorrectly classified as 0s. **False negatives**
# * 5528 were correctly classified as 0s. **True positives**
# 

# <img src="img\confusion matrix.jpg">

# # Precision
# 
# **Precision** measures the accuracy of positive predictions. Also called the `precision` of the classifier
# 
# $$\textrm{precision} = \frac{\textrm{True Positives}}{\textrm{True Positives} + \textrm{False Positives}}$$
# 
# <img src="img\precision.jpg">

# In[40]:


from sklearn.metrics import precision_score, recall_score

# Note the result here may vary from the video as the results from the confusion matrix are different each time you run it.

# In[41]:


precision_score(y_train_0, y_train_pred)  # 5618 / (574 + 5618)

# In[45]:


5618 / (574 + 5618)

# ## Recall
# 
# `Precision` is typically used with `recall` (`Sensitivity` or `True Positive Rate`). The ratio of positive instances that are correctly detected by the classifier.
# 
# $$\textrm{recall} = \frac{\textrm{True Positives}}{\textrm{True Positives} + \textrm{False Negatives}}$$
# 
# <img src="img\recall.jpg">

# Note the result here may vary from the video as the results from the confusion matrix are different each time you run it.

# In[46]:


recall_score(y_train_0, y_train_pred)  # 5618 / (305 + 5618)

# In[47]:


5618 / (305 + 5618)

# ## F1 Score
# 
# $F_1$ score is the harmonic mean of precision and recall. Regular mean gives equal weight to all values. Harmonic mean gives more weight to low values.
# 
# 
# $$F_1=\frac{2}{\frac{1}{\textrm{precision}}+\frac{1}{\textrm{recall}}}=2\times \frac{\textrm{precision}\times \textrm{recall}}{\textrm{precision}+ \textrm{recall}}=\frac{TP}{TP+\frac{FN+FP}{2}}$$
# 
# The $F_1$ score favours classifiers that have similar precision and recall.
# 

# In[48]:


from sklearn.metrics import f1_score

# Note the result here may vary from the video as the results from the confusion matrix are different each time you run it.

# In[49]:


f1_score(y_train_0, y_train_pred)

# ***

# # Precision / Recall Tradeoff
# 
# Increasing precision reduced recall and vice versa

# <img src="img\precision-recall.png">

# Our classifier is designed to pick up zeros.
# 
# 12 observations
# 
# ***
# 
# **Central Arrow**
# 
# Suppose the decision threshold is positioned at the central arrow: 
# * We get 4 true positives (We have 4 zeros to the right of the central arrow)
# * 1 false positive which is actually seven.
# 
# At this threshold, the **precision accuracy** is $\frac{4}{5}=80\%$
# 
# However, out of the 6 zeros, the classifier only picked up 4. The **recall accuracy** is $\frac{4}{6}=67\%$
# 
# ***
# 
# **Right Arrow**
# 
# * We get 3 true positives
# * 0 false positive
# 
# At this threshold, the **precision accuracy** is $\frac{3}{3}=100\%$
# However, out of the 6 zeros, the classifier only picked up 3. The **recall accuracy** is $\frac{3}{6}=50\%$
# 
# ***
# 
# **Left Arrow**
# 
# * We get 6 true positives
# * 2 false positive
# 
# At this threshold, the **precision accuracy** is $\frac{6}{8}=75\%$
# Out of the 6 zeros, the classifier picked up all 6. The **recall accuracy** is $\frac{6}{6}=100\%$
# 
# ***
# 
# 
# 

# In[50]:


clf = SGDClassifier(random_state=0)
clf.fit(X_train, y_train_0)

# In[51]:


y[1000]

# In[52]:


y_scores = clf.decision_function(X[1000].reshape(1, -1))
y_scores

# In[53]:


threshold = 0

# In[54]:


y_some_digits_pred = (y_scores > threshold)

# In[55]:


y_some_digits_pred

# In[56]:


threshold = 40000
y_some_digits_pred = (y_scores > threshold)
y_some_digits_pred

# In[57]:


y_scores = cross_val_predict(clf, X_train, y_train_0, cv=3, method='decision_function')

# In[58]:


plt.figure(figsize=(12, 8));
plt.hist(y_scores, bins=100);

# With the decision scores, we can compute precision and recall for all possible thresholds using the `precision_recall_curve()` function:

# In[59]:


from sklearn.metrics import precision_recall_curve

# In[60]:


precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)


# In[61]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([-0.5, 1.5])


# In[62]:


plt.figure(figsize=(12, 8));
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# With this chart, you can select the threshold value that gives you the best precision/recall tradeoff for your task.
# 
# Some tasks may call for higher precision (accuracy of positive predictions). Like designing a classifier that picks up adult contents to protect kids. This will require the classifier to set a high bar to allow any contents to be consumed by children.
# 
# Some tasks may call for higher recall (ratio of positive instances that are correctly detected by the classifier). Such as detecting shoplifters/intruders on surveillance images - Anything that remotely resemble "positive" instances to be picked up.
# 
# ***

# One can also plot precisions against recalls to assist with the threshold selection

# In[63]:


plt.figure(figsize=(12, 8));
plt.plot(precisions, recalls);
plt.xlabel('recalls');
plt.ylabel('precisions');
plt.title('PR Curve: precisions/recalls tradeoff');

# # Setting High Precisions
# 
# Let's aim for 90% precisions.

# In[64]:


len(precisions)

# In[65]:


len(thresholds)

# In[66]:


plt.figure(figsize=(12, 8));
plt.plot(thresholds, precisions[1:]);

# In[134]:


idx = len(precisions[precisions < 0.9])

# In[135]:


thresholds[idx]

# In[136]:


y_train_pred_90 = (y_scores > 21454)

# In[137]:


precision_score(y_train_0, y_train_pred_90)

# In[138]:


recall_score(y_train_0, y_train_pred_90)

# # Setting High Precisions
# 
# Let's aim for 99% precisions.

# In[139]:


idx = len(precisions[precisions < 0.99])

# This is the same as the line above
idx = len(precisions) - len(precisions[precisions > 0.99])
# In[145]:


thresholds[idx]

# In[146]:


y_train_pred_90 = (y_scores > thresholds[idx])

# In[147]:


precision_score(y_train_0, y_train_pred_90)

# In[148]:


recall_score(y_train_0, y_train_pred_90)

# #### Exercise
# 
# High Recall Score. Recall score > 0.9

# In[149]:


idx = len(recalls[recalls > 0.9])

# In[150]:


thresholds[idx]

# In[151]:


y_train_pred_90 = (y_scores > thresholds[idx])

# In[152]:


precision_score(y_train_0, y_train_pred_90)

# In[153]:


recall_score(y_train_0, y_train_pred_90)

# ***

# ## The Receiver Operating Characteristics (ROC) Curve

# Instead of plotting precision versus recall, the ROC curve plots the `true positive rate` (another name for recall) against the `false positive rate`. The `false positive rate` (FPR) is the ratio of negative instances that are incorrectly classified as positive. It is equal to one minus the `true negative rate`, which is the ratio of negative instances that are correctly classified as negative.
# 
# The TNR is also called `specificity`. Hence the ROC curve plots `sensitivity` (recall) versus `1 - specificity`.

# <img src="img\tnr_and_fpr.png">

# In[154]:


from sklearn.metrics import roc_curve

# In[155]:


fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)


# In[156]:


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')


# In[157]:


plt.figure(figsize=(12, 8));
plot_roc_curve(fpr, tpr)
plt.show();

# In[158]:


from sklearn.metrics import roc_auc_score

# In[159]:


roc_auc_score(y_train_0, y_scores)

# Use PR curve whenever the **positive class is rare** or when you care more about the false positives than the false negatives
# 
# Use ROC curve whenever the **negative class is rare** or when you care more about the false negatives than the false positives

# 
# In the example above, the ROC curve seemed to suggest that the classifier is good. However, when you look at the PR curve, you can see that there are room for improvement.

# # Model Comparison
# 
# # Random Forest

# In[160]:


from sklearn.ensemble import RandomForestClassifier

# In[161]:


f_clf = RandomForestClassifier(random_state=0)

# In[162]:


y_probas_forest = cross_val_predict(f_clf, X_train, y_train_0,
                                    cv=3, method='predict_proba')

# In[163]:


y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_0, y_scores_forest)

# In[164]:


plt.figure(figsize=(12, 8));
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show();

# In[165]:


roc_auc_score(y_train_0, y_scores_forest)

# In[166]:


f_clf.fit(X_train, y_train_0)

# In[167]:


y_train_rf = cross_val_predict(f_clf, X_train, y_train_0, cv=3)

# In[168]:


precision_score(y_train_0, y_train_rf)

# In[169]:


recall_score(y_train_0, y_train_rf)

# In[170]:


confusion_matrix(y_train_0, y_train_rf)

# ***
