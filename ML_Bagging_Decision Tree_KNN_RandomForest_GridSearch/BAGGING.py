import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score,roc_curve
from pandas_profiling import ProfileReport
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


df=pd.read_csv('https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-red.csv',
               delimiter=";")
print(df.head())
print(df.shape)
x=df.drop(['quality'],axis=1)
y=df['quality']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=5)
alg_bag=BaggingClassifier(DecisionTreeClassifier(),n_estimators=100)
alg_bag.fit(x_train,y_train)
y_pred=alg_bag.predict(x_test)
print(alg_bag.score(x_train,y_train))
print(alg_bag.score(x_test,y_test))
print(alg_bag.classes_)

alg_bag_Knn=BaggingClassifier(KNeighborsClassifier(n_neighbors=6),n_estimators=100)
alg_bag_Knn.fit(x_train,y_train)
y_pred2=alg_bag_Knn.predict(x_test)
print(alg_bag_Knn.score(x_train,y_train))
print(alg_bag_Knn.score(x_test,y_test))

alg_bag_ranforest=BaggingClassifier(RandomForestClassifier(),n_estimators=10,random_state=5)
alg_bag_ranforest.fit(x_train,y_train)
print(alg_bag_ranforest.score(x_train,y_train))
print(alg_bag_ranforest.score(x_test,y_test))

alg_ranforest= RandomForestClassifier()
alg_ranforest.fit(x_train,y_train)
print(alg_ranforest.score(x_train,y_train))
print(alg_ranforest.score(x_test,y_test))
print(alg_ranforest.estimators_)

#Let us print the tree plot of the random forest
plt.figure(figsize=(20,20))
tree.plot_tree(alg_ranforest.estimators_[0],filled=True)
plt.show()

print(alg_bag.estimators_)
plt.figure(figsize=(20,20))
tree.plot_tree(alg_bag.estimators_[0],filled=True)
plt.show()

print(alg_bag_Knn.estimators_)
# plt.figure(figsize=(20,20))
# tree.plot_tree(alg_bag_Knn.estimators_[0])
# plt.show()

grid_param={"n_estimators":[5,10,50,100,120,150],
            "criterion":['gini','entropy'],
            "max_depth":range(10),
            "min_samples_leaf":range(10),
            }

grid_cv=GridSearchCV(estimator=alg_ranforest, param_grid=grid_param,cv=6,n_jobs=-1,verbose=1)
grid_cv.fit(x_train,y_train)
best_params=grid_cv.best_params_
print(best_params)

#Let us re build the randomforest using the best parameters
alg_ranforest_new=RandomForestClassifier(n_estimators=120,criterion='entropy',max_depth=9,min_samples_leaf=1)
alg_ranforest_new.fit(x_train,y_train)
print(alg_ranforest_new.score(x_train,y_train))
print(alg_ranforest_new.score(x_test,y_test))

plt.figure(figsize=(20,20))
tree.plot_tree(alg_ranforest_new.estimators_[0],filled=True)
#tree.plot_tree(alg_bag.estimators_[0],filled=True)
plt.show()
