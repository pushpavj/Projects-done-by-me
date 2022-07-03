import pandas as pd
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

df=pd.read_csv('ai4i2020.csv')
print(df.head())
print(df.shape)

#profile=ProfileReport(df)
#profile.to_file('Report.html')


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

fig=pltl.box(df['Torque [Nm]'])
fig.write_html('Torque.html',auto_open=True)

q1=df['Torque [Nm]'].quantile(0.25)
q3=df['Torque [Nm]'].quantile(0.75)
iqr=1.5*(q3-q1)
fence_low=q1-iqr
fence_high=q3+iqr

df['new_Torque']=df[(df['Torque [Nm]']>fence_low) & (df['Torque [Nm]']<fence_high)]['Torque [Nm]']
print(df['new_Torque'])

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
from sklearn.model_selection import train_test_split

#Since 'Process temperature [K]' is in linear relation with Air Temperature
#First let us consider only 'Process temperature [K]' feature for our model
x=X[['Process temperature [K]']]
print(x)

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
#y_train_pred=alg_linear.predict(x_train)

y_dash=[]
for i in range(len(x_train)):
    y_dash.append(m*x_train.iloc[i][0]+c)
ydash=pd.DataFrame(y_dash)
yddash=ydash[0]
# plt.scatter(x,y)
# plt.scatter(x,y_dash)
#plt.show()

fig=pltl.scatter(x_train,y_train,trendline='ols')
#fig.add_line(x_train,y_dash)
fig.write_html('Lineargraph.html',auto_open=True)

fig=pltl.scatter(x_train,y_train)
fig.write_html('Train_scatter.html',auto_open=True)