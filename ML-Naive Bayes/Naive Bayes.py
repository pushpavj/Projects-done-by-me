import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score,roc_curve
import matplotlib.pyplot as plt


df=pd.read_csv('diabetes.csv')
print(df)

print(df.isnull().sum())

print(df.describe())

x=df.drop(columns='Outcome')
y=df.Outcome

alg_scaler= StandardScaler()
scaled_x=pd.DataFrame(alg_scaler.fit_transform(x))
print(scaled_x)
x_train,x_test,y_train,y_test=train_test_split(scaled_x,y,test_size=.20,random_state=30)
print(x_train)
print(y_test)
alg_naive=GaussianNB()
alg_naive.fit(x_train,y_train)
y_pred=alg_naive.predict(x_test)
print(alg_naive.score(x_test,y_test))
print(alg_naive.score(x_train,y_train))

#Check the VIF to find out the dependencies between the feature
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['vif']=[variance_inflation_factor(scaled_x,i) for i in range(scaled_x.shape[1])]
vif['Features']=x.columns
print(vif)
'''Observation the vif is very less i.e <10 so we can ignore '''