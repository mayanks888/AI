import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
print(train_data.head())
print('shape of the train_data is ', train_data.shape)

#lets remove id
train_data.drop("Id", axis = 1, inplace = True)

# features=train_data.iloc[:,0:-1].values
features=train_data.iloc[:,[1,4,-1]].values
label=train_data.iloc[:,-1].values
# print(features[7])

df=pd.DataFrame(features,label)




# most correlated features
corrmat = train_data.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(train_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# plt.show()
print(g)


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_data[cols], size = 1.5)
# plt.show()

ntrain = train_data.shape[0]
ntest = test_data.shape[0]
y_train = train_data.SalePrice.values
all_data = pd.concat((train_data, test_data)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()



'''corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
# plt.show()

plot_corr(df=df)
corr = df.corr()
corr.style.background_gradient()

def plot_corr(df,size=10):
    Function plots a graphical correlation matrix for each pair of columns in the train_dataframe.

    Input:
        df: pandas train_dataFrame
        size: vertical and horizontal size of the plot

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    # plt.show()'''
