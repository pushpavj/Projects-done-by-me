import numpy as np
import pandas as pd
import seaborn as sns
import os
import pandas_profiling

df=pd.read_csv('FitBit data.csv')
print(df.head())
print(df.profile_report())

profile=df.profile_report(title='Fitbit profile report')
profile.to_file(output_file='Fibbitprofile.html')
print(df.shape)
print(df.isnull().sum())
print(df.columns)
print(df.dtypes)
print(df["ActivityDate"].unique())

df['newdate']=pd.DatetimeIndex(df['ActivityDate'])
print(df.dtypes)
print(df['newdate'].head())
df['year']=pd.DatetimeIndex(df['ActivityDate']).year
df['month']=pd.DatetimeIndex(df['ActivityDate']).month
df['day']=pd.DatetimeIndex(df['ActivityDate']).day
df['dayname']=pd.DatetimeIndex(df['ActivityDate']).day_name()
df['monthname']=pd.DatetimeIndex(df['ActivityDate']).month_name()
print(df[['newdate','year','month','day','monthname','dayname']])
df.drop(['dayname'],axis=1,inplace=True)
print(df[['newdate','year','month','day','monthname',]])
import matplotlib.pyplot as plt
print(df['newdate'])
sns.boxplot(x='newdate',y='Calories',data=df)
#plt.figure(figsize=(10,5))
plt.savefig('boxplt.png')
plt.show()
boxplt=sns.boxplot(x='newdate',y='Calories',data=df)

df['week of the year']=pd.DatetimeIndex(df['ActivityDate']).week
print(df[['week of the year','newdate']])
sns.boxplot(x='week of the year',y='Calories',data=df)
plt.show()
sns.boxplot(x='month',y='Calories',data=df)
plt.show()
sns.boxplot(x='Calories',y='TotalSteps',data=df)
plt.show()