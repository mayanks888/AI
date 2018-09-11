from sklearn import datasets
from sklearn.model_selection import train_test_split


boston=datasets.load_boston()
print(boston.data.shape)
print(boston.target.shape)
print(boston.data[1:6,:])


#learn more about cross validation
from sklearn.model_selection import KFold
kf=KFold(n_splits=5)#split the data into 5 ramdom parts
for train,test in kf.split(boston.data):
    print("the spliting is {sp}and {tf}".format(sp=train.shape, tf=test.shape))


from sklearn.model_selection import StratifiedKFold
kf=StratifiedKFold(n_splits=3)#split the data into 5 ramdom parts
for train,test in kf.split(boston.data,boston.target):
    print("the spliting is {sp}and {tf}".format(sp=train.shape, tf=test.shape))