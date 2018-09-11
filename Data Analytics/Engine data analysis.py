import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data=pd.read_excel("EngRpm.XLS")

new_data=data[['Gr[]','WhlRPM_FL[rpm]','EngRPM[rpm]','AccelPdlPosn[%]','EngTrq[Nm]','EngRPM[rpm]']]#to print new column of your choice
data.dropna(inplace=False)



# print new_data[(new_data['Gr[]'].notnull())|(new_data['WhlRPM_FL[rpm]'].notnull())].head(500)
data= new_data[(new_data['WhlRPM_FL[rpm]'].notnull())]
# print datasets[(datasets['Pclass']>1) & (datasets['Age']>25)].head()
# print new_data ['Gr[]'].isnull().sum()
# print new_data ['WhlRPM_FL[rpm]'].isnull().sum()
# print data['WhlRPM_FL[rpm]'].notnull().sum()
# print data['Gr[]'].notnull().sum()

# new_data=new_data.replace(np.nan,'pizza')

data=data.fillna(method='ffill')
print (data.head(20))
print (data.info())
print (data.shape)
data.to_csv("MY_refine_engine_data", sep='\t')
# data.to_excel("MY_refine_engine_data1",'Sheet1')

writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
data.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()