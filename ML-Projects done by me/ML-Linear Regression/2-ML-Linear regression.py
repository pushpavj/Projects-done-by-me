import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pandas_profiling import ProfileReport

df=pd.read_csv('Advertising.csv')

print(df.head())
print(df.shape)

#Problem statement1: If I am investing these many  advertisements in TV radio and Newpaper then build a
# model on my sales i.e. build a model to predict the sales
#Proeblem statment2: Among the TV readio and newspaper which kind of advertisement is doing better
#if there are any relationship between the data sets then showcase the same

profile=ProfileReport(df)
profile.to_file('Report.Html')
x=df[['TV']]
y=df["sales"]

from sklearn.linear_model import LinearRegression
alg_linear=LinearRegression()
alg_linear.fit(x,y)
print(alg_linear.intercept_, alg_linear.coef_)
file=('Liear_reg_model.csv')
pickle.dump(alg_linear,open(file,'wb'))
print(alg_linear.predict([[45]]))
l=[2,3.4,5,5,6,6,6,89]

for i in l:
    print(alg_linear.predict([[i]]))


saved_model=pickle.load((open(file,'rb')))
print(saved_model.predict([[45]]))

print(alg_linear.score(x,y))

data_x=df[['TV','radio','newspaper']]
y=df.sales
alg_lm=LinearRegression()
alg_lm.fit(data_x,y)
print(alg_lm.intercept_,alg_lm.coef_)
print(alg_lm.score(data_x,y))