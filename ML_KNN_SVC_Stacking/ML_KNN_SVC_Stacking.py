import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from pandas_profiling import ProfileReport
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df=pd.read_csv('https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-red.csv',
               delimiter=";")
print(df.head())
print(df.shape)
x=df.drop(['quality'],axis=1)
y=df['quality']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=5)

alg_Knn=KNeighborsClassifier()
alg_Knn.fit(x_train,y_train)
print(alg_Knn.score(x_train,y_train))
print(alg_Knn.score(x_test,y_test))

grid_param={'n_neighbors':[3,5,7,9,12,13,14,15,17,21],'algorithm':['auto','ball_tree','kd_tree','brute'],
            'leaf_size':[10,15,20,25,30,35,40,45,50],
            'p':[1,2], 'weights':['uniform','distance']}
grid_cv=GridSearchCV(estimator=alg_Knn,param_grid=grid_param,n_jobs=-1)
grid_cv.fit(x_train,y_train)
print(grid_cv.best_params_)

#Let us rebuild the model with best parameter and see

alg_Knn_new=KNeighborsClassifier(n_neighbors=14,p=1,weights='distance',algorithm='auto',leaf_size=10)
alg_Knn_new.fit(x_train,y_train)
print(alg_Knn_new.score(x_train,y_train))
print(alg_Knn_new.score(x_test,y_test))

import pickle
pickle.dump(alg_Knn_new,open('Knn.pkl','wb'))

#Let us perform SVC support vecotor classifier
alg_svc=SVC()
alg_svc.fit(x_train,y_train)
print(alg_svc.score(x_train,y_train))
print(alg_svc.score(x_test,y_test))

#Let us get the best parameters from Grid search CV

grid_param={"kernel":['linear', 'poly', 'rbf', 'sigmoid']
#            "C":[.1,.4,.6,1,2,3,4,100,200,500],
  #          "gamma":[.001,.4,.003,.1,.004],


            }
grid_cv=GridSearchCV(param_grid=grid_param,estimator=alg_svc,verbose=True)
grid_cv.fit(x_train,y_train)
print(grid_cv.best_params_)


#Let us understand SVR

df2=pd.read_csv('Admission_Prediction.csv')
print(df2.head())
print(df2.columns)

print(df2.isnull().sum())
df2['GRE Score']=df2['GRE Score'].fillna(df2['GRE Score'].mean())
df2['TOEFL Score']=df2['TOEFL Score'].fillna(df2['TOEFL Score'].mean())
df2['University Rating']=df2['University Rating'].fillna(df2['University Rating'].mean())
x=df2.drop(columns=['Serial No.','Chance of Admit'])
y=df2['Chance of Admit']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30,test_size=.20)

alg_svr=SVR()
alg_svr.fit(x_train,y_train)
print(alg_svr.score(x_train,y_train))
print(alg_svr.score(x_test,y_test))
y_predict=alg_svr.predict(x_test)
print(r2_score(y_test,y_predict))


#Let us perform stacking
from sklearn.ensemble import _stacking

df3=pd.read_csv('https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-red.csv',
               delimiter=";")
print(df3.head())
print(df3.shape)

x=df3.drop(columns=['quality'])
y=df3['quality']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=30)

alg_knn=KNeighborsClassifier()
alg_knn.fit(x_train,y_train)
print(alg_knn.score(x_test,y_test))

alg_svc=SVC()
alg_svc.fit(x_train,y_train)
print(alg_svc.score(x_test,y_test))

x_sub,x_for_stack,y_sub,y_stack=train_test_split(x,y,test_size=.50,random_state=30)

x_sub_train,x_sub_test,y_sub_train,y_sub_test=train_test_split(x_sub,y_sub,test_size=.20,random_state=30)

alg_knn_sub=KNeighborsClassifier()
alg_knn_sub.fit(x_sub_train,y_sub_train)

alg_svc_sub=SVC()
alg_svc_sub.fit(x_sub_train,y_sub_train)

x1_feature=alg_knn_sub.predict(x_for_stack)
x2_feature=alg_svc_sub.predict(x_for_stack)

stack_features=np.column_stack((x1_feature,x2_feature))
stack_y=y_stack

#Let us consider the random forest as stacking 3rd algorithm which uses the predicted out put of algorithm1 and
#algorithm2 as its input features.

alg_stack= RandomForestClassifier()
alg_stack.fit(stack_features,stack_y)

#To predict we need the data in the same format as in stack_features, so let us make the data to pass trhogh
#the same pipe line as we did for getting training data for stack algorithm

x1_test_feature=alg_knn_sub.predict(x_sub_test)
x2_test_feature= alg_svc_sub.predict(x_sub_test)

stack_test_features=np.column_stack((x1_test_feature,x2_test_feature))

stack_predict=alg_stack.predict(stack_test_features)

print(alg_stack.score(stack_test_features,y_sub_test))
