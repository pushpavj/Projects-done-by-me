import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as pltl
from pandas_profiling import ProfileReport
df1=pd.read_csv('Home_Credit_Default_Risk1.csv')

# profile1=ProfileReport(df1)
# profile1.to_file('Home_profile.html')
df2=pd.read_csv('titanic.csv')

# profile2=ProfileReport(df2)
# profile2.to_file("profilrepot.html")

print('shape1',df1.shape)
print('shape2', df2.shape)


print('df1 is null sum',df1.isnull().sum())
print('df2 is null sum',df2.isnull().sum())

print('df1 head',df1.head())
print('df2.head',df2.head())

print('df1.info',df1.info())
print('df2.info', df2.info())


fig=pltl.imshow(df1)
fig.write_html('df1heatmap.html',auto_open=True)

fig=pltl.imshow(df2)
fig.write_html('df2heatmap.html',auto_open=True)

print('df2.cols',df2.columns)
#
fig=pltl.imshow(df2[['Survived']],text_auto=True)
fig.write_html('Survived.html',auto_open=True)

# fig=pltl.imshow()

import seaborn as sns
sns.heatmap(df2[['Survived']])
plt.show()


import seaborn as sns
sns.heatmap(df2[['Survived','Pclass']])
plt.show()

fig=pltl.imshow(df2[['Survived','Pclass','Sex']],text_auto=True)
fig.write_html('Survived.html',auto_open=True)

# import seaborn as sns
# sns.heatmap(df2[['Survived','Pclass','Sex']])
# plt.show()
# import seaborn as sns
# sns.heatmap(df2[['Sex']])
# plt.show()

fig=pltl.imshow(df2[['Age']])
fig.write_html('Age.html',auto_open=True)
sns.heatmap(df2[['Age']])
plt.show()

import missingno as msno
msno.heatmap(df2)
plt.show()

print(df2.isnull().sum())


msno.bar(df2)
plt.show()

msno.dendrogram(df2)
plt.show()
# We got to know that columns Age, Cabin and Embarked are having missing values now we will
#need to handle these missing values
print(df2[df2["Age"].isnull()].index)
print(df2[["Age"]].isnull())
print(list(df2[df2["Age"].isnull()].index))
print(df2['Age'].iloc[df2[df2["Age"].isnull()].index])

#print(df2['Age'].fillna(0,inplace=True))
print(df2['Age'].fillna(0))
print(df2["Age"].isnull().sum())

#function to handle missiing value with filling it with any random value
def miss_fill(df,variable,randnum):
    df[variable+'randnum']=df[variable].fillna(randnum)
    return df
miss_fill(df2,'Age','200')
print(df2.columns)
print(df2['Agerandnum'])

miss_fill(df2,'Cabin',2)
print(df2.columns)
print(df2['Cabinrandnum'])

#forward fill and backword fill method for handling missing values
#Forward fill: How does this works is it will move forward to find the missing value
# once it finds it will fill it with its previous value.
#Backward fill: How does this works in it will move towards forwad to find the
# missing value but once it finds it it will fill it with its next value.


def miss_fill1(df,variable):
    df[variable+'forward']=df[variable].fillna(method='pad')
    df[variable+'backward']=df[variable].fillna(method='backfill')

miss_fill1(df2,'Age')
print(df2.columns)
print(df2['Ageforward'])
print(df2['Agebackward'])
# Another way of handling missing value is dropping the missing value observations
# or records by using dropna:
#
# Dropna will drops the rows having missing value and drops the column having
#missing value

#Drops all the rows which has missing value
print(df2.dropna(axis=0))
#Drops all the columns which have missing value
print(df2.dropna(axis=1))

print(df2['Age'].dropna(axis=0))

#by specifying the tresh we can control the scinario of missing rows or columns to be deleted
# with thresh it looks for the condition of macthing thresh before dropping it will drop only
# when the thresh condition is not satisfied. Here in this example the scinario is do not drop
#if at least 15 columns have non null value otherwise drop that missing row.i.e. it will drop
# the row if less than 15 columns have non null value. In this case out of 16 column if the
#row has missing value in only one column then it will not drop that row. In case if there
#are 2 or more column have missing value then it will drop that row.
#Similarly thresh is also applicable for column as well
print(df2.dropna(axis=0,thresh=15))

print(df2.dropna(axis=1,thresh=800))

#Random sample imputaion method for handling missing value.
#How does this work is: For example a column X has total 500 records out of which 168 are
# missing then, as part of random sample imputation method first it takes 168 samples from
# the total records 500 and using this sampled data it will replace the 168 missing value
# field.

#Let us take a random sample from the Age column after dropping all the na values
random_sample=df2['Age'].dropna().sample(df2['Age'].isnull().sum(), random_state=0)
print('random_sample', random_sample)
print(type(random_sample))
#Set the index of the random_sample same as missing value index
random_sample.index=df2[df2['Age'].isnull()].index
#Create new age column with same as Age
df2['Age_random']=df2['Age']
#Assign the random_sample values to the new age column at index where Age is null
df2.loc[df2["Age"].isnull(), 'Age_random']=random_sample
#in the above code the df2["Age"].isnull() gives the boolean list of true or false
#so df.loc[row name index,column nameindex] so the boolean list gives the row index
#where the missing Age value present and the column name "Age_random" says the column
#to be selected
print(df2.loc[df2['Age'].isnull(),'Age_random'])

# we can also wirte a function for random sample imputation
def rand_imp(df,variable):
    df[variable+'rand']=df[variable]
#    random_sample = df2['Age'].dropna().sample(df2['Age'].isnull().sum(), random_state=0)
    rand_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    rand_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable+'rand']=rand_sample
rand_imp(df2,'Age')
print(df2.loc[df2['Age'].isnull(),'Agerand'])
rand_imp(df2,'Embarked')
print(df2.loc[df2['Embarked'].isnull(),'Embarkedrand'])

#Another way of handling missing value is filling it with mean median and mode value

def fill_mean(df,variable,varmean):
    df[variable+'mean']=df[variable].fillna(varmean)

varmean=df2['Age'].dropna().mean()
fill_mean(df2,'Age',varmean)
print(df2.loc[df2["Age"].isnull(),"Agemean"])

def fill_median(df,variable,varmedian):
    df[variable+'median']=df[variable].fillna(varmedian)

varmedian=df2["Age"].median()
fill_median(df2,'Age',varmedian)
print(df2.loc[df2["Age"].isnull(),"Agemedian"])

def fill_mode(df,variable,varmode):
    df[variable+'mode']=df[variable].fillna(varmode)

varmode=df2["Age"].dropna().mode()
print('varmode=',varmode[0])
print(type(varmode))
fill_mode(df2,"Age",varmode[0])
print(df2.loc[df2["Age"].isnull(),"Agemode"])

#Another way of handling missing value is Capturing NAN vlaue with new feature data
import numpy as np
df2['Age_Nan']=np.where(df2["Age"].isnull(),1,0) # 1 will replace the True and 0 will
# replace the False. Here new feature is created with having a value 1 for missing value
# and value zero for rest of the entire observations.

print(df2.loc[df2['Age'].isnull(),'Age_Nan'])
print(df2[['Age','Age_Nan']].head(20))

#Another way of handling missing value is End of distribution imputation
#i.e Here we are going to calculate the 3rd standard diviation value and filling the
# missing value with these extreme 3rd standard diviation value
extreame=df2["Age"].mean()+3*df2['Age'].std()
df2['Age_end_dist']=df2["Age"].fillna(extreame)
print('Mean',df2['Age'].mean())
print('3rd Std=',3*df2['Age'].std())
print('extreame',extreame)
print(df2.loc[df2['Age'].isnull(),'Age_end_dist'])
df2['Age'].hist(bins=50)
plt.show()
df2['Age_end_dist'].hist(bins=50)
plt.show()

#Another way of handling missing value is building a ML model to predict the missing value
#form the x_train and y_train features
print(df2.columns)
df3=df2[['Survived','Pclass','Fare','Age']].dropna()
print(df3.head())
x_train=df3.drop('Age',axis=1)
y_train=df3['Age']
print(x_train.head())
print(y_train.head())
#form the x_test feature as x fetaures where missing Age value
x_test=df2.loc[df2['Age'].isnull(),['Survived','Pclass','Fare']]
#Build the model
from sklearn.linear_model import LinearRegression
alg_lin=LinearRegression()
alg_lin.fit(x_train,y_train)
#Predict the missing age
y_pred=alg_lin.predict(x_test)
print('x_test',x_test)
print('y_pred',y_pred)
#Assign the predicted value to new Age feature column
df2['Age_predict']=df2["Age"]
ypred=pd.Series(y_pred)
print('ypred',ypred)
print('ypred_ind',ypred.index)
ypred.index=df2[df2["Age"].isnull()].index
print('ypred_index',ypred.index)
df2.loc[df2['Age'].isnull(),'Age_predict']=ypred
print(df2[["Age","Age_predict"]])
print(df2.loc[df2['Age'].isnull(),'Age_predict'])