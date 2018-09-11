import pandas as pd
# now we will modify our data into categorical variable and transform it

data= pd.read_csv("dataprepossor.csv")
#this is to convert into dummy categorical sheets
data2=pd.get_dummies(data['Country'])
print data2

#categorical numerical and disrete
data= pd.read_csv("train.csv")
# categorical data="sex, survived , embarked", its is basically one out of given choice
#continous data= age,fare,any random data can be possible
# discrte data= sibsb, parch, and some definate discrete value is possible
# ordinal-->'P class , both categorical and numeric(continuous)

print (data.describe(include=['O'])#its catergorises the given data)
print (data.describe(include='all')#its catergorises the all data)