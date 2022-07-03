import pandas as pd
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

df=pd.read_csv('ai4i2020.csv')
print(df.head())
print(df.shape)


#########EDA START#######################################################################
profile=ProfileReport(df)
profile.to_file('Report.html')


#UDI is having serial numbers only, hence drop the UDI column

print(df.columns)
df.drop('UDI',axis=1, inplace=True)
print(df.head())
print(df['Product ID'].unique)
print(pd.value_counts(df['Product ID']))
#print(len(print(pd.value_counts(df['Product ID']))))
print(df['Product ID'].isna())
print(sum(df['Product ID'].isna()))
#plt.hist(df['Type'])
#plt.show()
print(df.duplicated()) #finds the duplicate columns if any
print(sum(df.duplicated()))
#Product ID is having serial numbers only, hence drop the UDI column, The type column has M,L,H grouping
# data so product id is not required.

#plt.hist(df['Product ID'])
#plt.show()
dups=df[['Product ID']].duplicated() #to find the duplicates inside the coulmn
print('dups',sum(dups))  #to get the number of duplicates
df.drop('Product ID',axis=1,inplace=True)
print(df.head())

import plotly.express as pltl

print(dir(pltl))



fig=pltl.histogram(df['Type'])
fig.write_html('first_figure.html', auto_open=True) #stores graph in to html file and also opens the file during executeion

obj=df.describe()
print(obj)


df.info()  #This automatically prints without print statement

obj.to_csv('describe.csv')

fig=pltl.box(df['Process temperature [K]'])
fig.write_html('Process temperature.html',auto_open=True)

fig=pltl.box(df['Rotational speed [rpm]'])
fig.write_html('Rotational speed.html',auto_open=True)
#************************************EDA_END#########################################
#********************************Outlier Treatment Start*****************************
q1=df['Rotational speed [rpm]'].quantile(0.25)
q3=df['Rotational speed [rpm]'].quantile(0.75)
iqr=1.5*(q3-q1)
fence_low=q1-iqr
fence_high=q3+iqr
print(df['Rotational speed [rpm]']>fence_low)
print((df['Rotational speed [rpm]']>fence_low) & (df['Rotational speed [rpm]']<fence_high))
print(df[(df['Rotational speed [rpm]']>fence_low) & (df['Rotational speed [rpm]']<fence_high)].index)
df['new_Rotational speed']=df[(df['Rotational speed [rpm]']>fence_low) & (df['Rotational speed [rpm]']<fence_high)]['Rotational speed [rpm]']
print(df['new_Rotational speed'])


fig=pltl.box(df['new_Rotational speed'])
fig.write_html('new_rotational.html',auto_open=True)
#
fig=pltl.box(df['Torque [Nm]'])
fig.write_html('Torque.html',auto_open=True)

q1=df['Torque [Nm]'].quantile(0.25)
q3=df['Torque [Nm]'].quantile(0.75)
iqr=1.5*(q3-q1)
fence_low=q1-iqr
fence_high=q3+iqr

df['new_Torque']=df[(df['Torque [Nm]']>fence_low) & (df['Torque [Nm]']<fence_high)]['Torque [Nm]']
print(df['new_Torque'])
#
fig=pltl.box(df['new_Torque'])
fig.write_html('new_Torque.html',auto_open=True)

df.drop(['Torque [Nm]','Rotational speed [rpm]'],axis=1,inplace=True)
print(df.head())

fig=pltl.box(df['Tool wear [min]'])
fig.write_html('Tool wear.html',auto_open=True)

fig=pltl.box(df['Machine failure'])
fig.write_html('Machine failure.html',auto_open=True) #categorical with value only o and 1s
fig=pltl.box(df['TWF'])
fig.write_html('TWF.html',auto_open=True) #categorical with value only o and 1s

fig=pltl.scatter(data_frame=df, y='Air temperature [K]',x='Process temperature [K]')
fig.write_html('scatter3d.html',auto_open=True)
#**************************Out lier treatment End*************************************************
#******************Data pre processing start**************************************
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(handle_unknown='ignore')
encode_df=pd.DataFrame(encoder.fit_transform(df[['Type']]).toarray())
df2=df.join(encode_df)
print(df2.head())

df2.drop('Type',axis=1,inplace=True)

print(df2.head())
print(df2[['Air temperature [K]']])
X=df2
y=df2['Air temperature [K]']
X.drop('Air temperature [K]',axis=1, inplace=True)


print(X.head())
print(y.head())
print(X.columns)
#**********************Data preprocessing End***************************************************
#******************************model building start********************************************
from sklearn.model_selection import train_test_split
#
# Since 'Process temperature [K]' is in linear relation with Air Temperature
# First let us consider only 'Process temperature [K]' feature for our model
x=X[['Process temperature [K]']]
#***********************************************Model-1 - start************************************
df3=x.join(y)
print('df3',df3.head())

fig=pltl.scatter(data_frame=df3, x='Process temperature [K]',y='Air temperature [K]')
fig.write_html('df3.html',auto_open=True)

print('duplicated',df3.duplicated())
print('sumduplicate',sum(df3.duplicated()))
print('duplicate index',df3[df3.duplicated()].index)
print(df3.drop(df3[df3.duplicated()].index, axis=0, inplace=True))
print('duplicated sum',sum(df3.duplicated()))
print('shape of df3',df3.shape)
x1=df3[['Process temperature [K]']]
print('shape of x1',x1.shape)
y1=df3['Air temperature [K]']
print('shape of y1',y1.shape)
print('x1.head',x1.head(10))
print('y1.head',y1.head(10))
print('x1 type',type(x1))
print('y1 type', type(y1))

fig=pltl.scatter(x1,y1)
fig.write_html('nosplit.html',auto_open=True)
from sklearn.linear_model import LinearRegression
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,random_state=32)

fig=pltl.scatter(x1_train,y1_train)
fig.write_html('withsplit.html',auto_open=True)
alg_lin=LinearRegression()
alg_lin.fit(x1_train,y1_train)
y1_pred=alg_lin.predict(x1_test)

fig=pltl.scatter(x1_test,y1_pred)
fig.write_html('y1_pred.html',auto_open=True)

print('y1_coeff',alg_lin.coef_)
print('y1_intercpet',alg_lin.intercept_)
print('y1_score',alg_lin.score(x1_train,y1_train))
from sklearn.model_selection import cross_val_score

y_score=cross_val_score(alg_lin,x1_train,y1_train,cv=3)

print(y_score)
print('model-1 ends')
#*****************************************Model-1-end***********************************************
#*****************************************MOdel-2-start**************************************

x=X[['Process temperature [K]']]



x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=32)
from sklearn.linear_model import LinearRegression
alg_linear=LinearRegression()
alg_linear.fit(x_train,y_train)
y_pred=alg_linear.predict(x_test)
print(type(y_test))
print(type(y_pred))
print('test',type(y_test.iloc[0]))
print('pred',type(y_pred[0]))
num_correct=sum(y_pred==y_test)  #we can not calculate the num_correct like this as it is
# not a classification where we can find the exact match. But in continuous
# variable we will not be able to find the exact match of test with pred

acc=num_correct/len(y_test)
print('accuracy',acc)
print('Coefficient',alg_linear.coef_)
print('Intercept',alg_linear.intercept_)
print("score",alg_linear.score(x_train,y_train))
#When you’re applying .score(), the arguments are also the predictor x and response y,
# and the return value is 𝑅².

m=alg_linear.coef_
c=alg_linear.intercept_
y_train_pred=alg_linear.predict(x_train)

y_dash=[]
for i in range(len(x_train)):
    y_dash.append(m*x_train.iloc[i][0]+c)
ydash=pd.DataFrame(y_dash)
yddash=ydash[0]
plt.scatter(x,y)
plt.scatter(x_train,y_dash)
plt.show()

fig=pltl.scatter(x_train,y_train,trendline='ols')
#fig.add_line(x_train,y_dash)
fig.write_html('Lineargraph.html',auto_open=True)

fig=pltl.scatter(x_train,y_train)
fig.write_html('Train_scatter.html',auto_open=True)
from sklearn.model_selection import cross_val_score

y_score=cross_val_score(alg_linear,x_train,y_train,cv=3)

print(y_score)
print('model-2 ends')
#**************************************Model-2-end**********************************************
# *************************************Modle-3-start*******************************************
from sklearn.linear_model import LinearRegression
print('is na',X.isna())
print(X.columns)
print('is na sum',sum(X['Process temperature [K]'].isna()))
print('is na sum',sum(X['Tool wear [min]'].isna()))
print('is na sum',sum(X['Machine failure'].isna()))
print('is na sum',sum(X['TWF'].isna()))
print('is na sum',sum(X['HDF'].isna()))
print('is na sum',sum(X['PWF'].isna()))
print('is na sum',sum(X['OSF'].isna()))
print('is na sum',sum(X['RNF'].isna()))
print('is na sum',sum(X['new_Rotational speed'].isna()))
print('is na sum',sum(X['new_Torque'].isna()))

print('mean of new_torque',(X['new_Torque'].mean()))
print('mean of new_Rotational speed',(X['new_Rotational speed'].mean()))

X['new_Torque'].fillna(X['new_Torque'].mean(),inplace=True)
print('is na sum',sum(X['new_Torque'].isna()))
X['new_Rotational speed'].fillna(X['new_Rotational speed'].mean(), inplace =True)

print('is na sum',sum(X['new_Rotational speed'].isna()))


x2_train,x2_test,y2_train,y2_test=train_test_split(X,y,random_state=32)


alg_lin2=LinearRegression()
alg_lin2.fit(x2_train,y2_train)
y2_pred=alg_lin2.predict(x2_test)
print('y2_coeff',alg_lin2.coef_)
print('y2_intercept',alg_lin2.intercept_)
print('y2_score',alg_lin2.score(x2_train,y2_train))
print('modle-3 ends')
##############################Model-3- End********************************************
#*********************************Model valication starts*****************************
from sklearn.model_selection import cross_val_score

y_score=cross_val_score(alg_lin2,x2_train,y2_train,cv=3)

print(y_score)

print('Execution Successfully Completed')
###############Modle Validation ends***************************************************

















