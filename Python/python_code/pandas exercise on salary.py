import pandas as  pd
import numpy as np

df=pd.read_csv("Salaries.csv")
print(df.head())
print (df.info())

print ('the average basepay is :',df['BasePay'].mean())
print ('the maximum overtime pay is: ',df['OvertimePay'].max())
print ( 'the job tile of CHRISTOPHER CHONG is :', df[df['EmployeeName']=='CHRISTOPHER CHONG']['JobTitle'])
print ( 'the total that GARY JIMENEZ   make is {} :'.format( df[df['EmployeeName']=='GARY JIMENEZ']['TotalPay']))
# print ('the highest earning of person name is {}'.format(df[df['TotalPay'].argmax][['EmployeeName']]))
max_payrow = (df['TotalPay'].argmax())

# print (df['EmployeeName'][max_payrow])
print ('the highest earning i.e {earning} of person name is {name} '.format(name=df['EmployeeName'][max_payrow], earning=df['TotalPay'].max()))


new_data= (df.groupby('Year'))
# print (new_data.head())
print ('the mean of base pay is ', new_data['BasePay'].mean())

print ('the top five jobs are',df['JobTitle'].value_counts().head())
#
# print (new_data.describe())

# print (df[df['JobTitle'].value_counts()<2])
# print ( (df['JobTitle'].value_counts()==1)&(df['Year']==2013).sum()     )
print ((df['Year']==2013).sum())
print ( (df[df['Year']==2013]['JobTitle'].value_counts()==1).sum())
# print (my_new_data.value_count())