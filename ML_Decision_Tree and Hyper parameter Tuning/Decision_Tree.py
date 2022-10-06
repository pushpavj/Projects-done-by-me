import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,roc_curve
from pandas_profiling import ProfileReport as profile
df=pd.read_csv('https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-red.csv',
               delimiter=";")
print(df.head())
print(df.shape)
print(set(df['quality']))
# report=profile(df)
# report.to_file('wine_profile.html')

print(df.columns)

x=df.drop(['quality'],axis=1)
y=df['quality']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
alg_decision_gini=DecisionTreeClassifier()
alg_decision_gini.fit(x_train,y_train)
y_pred=alg_decision_gini.predict(x_test)
print(y_pred)
print('model accuracy',alg_decision_gini.score(x_test,y_test))
print(accuracy_score(y_test,y_pred))
print(y_test)
print('confusion_matrix \n',confusion_matrix(y_test,y_pred))

alg_decision_entropy=DecisionTreeClassifier(criterion='entropy')
alg_decision_entropy.fit(x_train,y_train)
print(alg_decision_entropy.score(x_test,y_test))
outfile=open('dt_entropy_meta.dot','w')
tree.export_graphviz(alg_decision_entropy,out_file=outfile,feature_names=x.columns)
print(dir())

import sklearn
import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(figsize=(20,20))
tree.plot_tree(alg_decision_entropy,filled =True)
plt.show()

df1=df.head(1000)
#x_df1=df1.drop(columns='quality')
x_df1=x_train
y_df1=y_train
#y_df1=df1['quality']
x1_train,x1_test,y1_train,y1_test=train_test_split(x_df1,y_df1)
alg_decision_gini.fit(x1_train,y1_train)
plt.figure(figsize=(10,10))
tree.plot_tree(alg_decision_gini,filled=True,class_names=[str(i) for i in set(y1_train)],feature_names=x_df1.columns)
plt.show()

alg_decision_gini.fit(x_df1,y_df1)
plt.figure(figsize=(10,10))
tree.plot_tree(alg_decision_gini,filled=True,class_names=[str(i) for i in set(y_df1)],feature_names=x_df1.columns)
plt.show()

path=alg_decision_gini.cost_complexity_pruning_path(x_df1,y_df1)
print(path)
ccpAlpha=path['ccp_alphas']
print(ccpAlpha)
#ccpAlpha=ccpalpha*10


dt_model_list=[]
for i in ccpAlpha:
    dt_m=DecisionTreeClassifier(ccp_alpha=i)
    dt_m.fit(x_df1,y_df1)
    dt_model_list.append(dt_m)
print(dt_model_list)

train_score=[i.score(x_df1,y_df1)for i in dt_model_list]
print(train_score)

test_score=[i.score(x_test,y_test) for i in dt_model_list]
print(test_score)

#Having the ticks to mark a line from x to y and y to x at certain points
fig,ax=plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy vs alpha for training and testing sets')
ax.plot(ccpAlpha,train_score,marker='o',label='train',drawstyle='steps-post')
ax.plot(ccpAlpha,test_score,marker='o',label='test',drawstyle='steps-post')
plt.grid(axis='both',)
plt.xticks(ticks=[0.00502,0.01515,0.02025,0.02535,0.03045,0.03555,0.04065,0.04575,0.05085,0.05595])
plt.yticks(ticks=[.40,.45,.50,.5512,.6013,.6414,.7015,.75616,.80,.85,.90,.952])
plt.ticklabel_format(axis='both')
ax.legend()
plt.show()

#After looking in to the graph selected couple of alpha value to tune to the point where both
#train score and test score coming clouser at maximum percentage.
alpha_list=[0.00602,0.00515,0.00525,0.00535,0.00545,0.00555,0.00565,0.00575,0.00585,0.00595]
for i in alpha_list:
    dt_m = DecisionTreeClassifier(ccp_alpha=i)
    dt_m.fit(x_df1, y_df1)
    print('alpha= ', i)
    print('train score',dt_m.score(x_df1,y_df1))
    print('test score',dt_m.score(x_test,y_test))



plt.figure(figsize=(10,10))
tree.plot_tree(dt_m,filled=True,class_names=[str(i) for i in set(y_df1)],feature_names=x_df1.columns)
plt.show()

#alg_dt1=DecisionTreeClassifier()
#
# Now we can find many parameters available under DicsionTree algorithm which we can control to get the better
# performing model. But how ever finding the best value for each parameter manually is a tedious job. So we have
# grid_search_cv and random_search_cv inbuilt methods to find the best parameters automatically.
# (CV-Cross validation). Both grid_search_cv and random_search_cv internally does the same thing as we did
# manually for finding the best fit parameters, it splits the data training and test repeatedly and applies
# the different different possible parameter values to it and checks the performance of the model and finally
# it will list the best fit parameter which gives the better performing model, even for multiple parameter
# at a time.
#
#
# Following are the parameters available in the DecisionTree algorithm
#
# def __init__(self,
#              *,
#              criterion: Any = "gini",
#              splitter: Any = "best",
#              max_depth: Any = None,
#              min_samples_split: Any = 2,
#              min_samples_leaf: Any = 1,
#              min_weight_fraction_leaf: Any = 0.0,
#              max_features: Any = None,
#              random_state: Any = None,
#              max_leaf_nodes: Any = None,
#              min_impurity_decrease: Any = 0.0,
#              class_weight: Any = None,
#              ccp_alpha: Any = 0.0) -> None
#

grid_param={ "criterion":[ "gini","entropy"],
             "splitter":["best","random"],
             "max_depth":range(2,40,1),
             "min_samples_split":range(2,10,1),
             "min_samples_leaf" : range(1,10,1)}
             # "random_state":range(0,10,1),
             #"ccp_alpha":np.linspace(0.0,1.0,30)
             #
             # }


             # min_weight_fraction_leaf: Any = 0.0,
             # max_features: Any = None,
             # random_state: Any = None,
             # max_leaf_nodes: Any = None,
             # min_impurity_decrease: Any = 0.0,
             # class_weight: Any = None,
             # ccp_alpha: Any = 0.0}
             #

grid_cv=GridSearchCV(estimator=dt_m,param_grid=grid_param,cv=5,n_jobs=-1)
#n_jobs=-1 (Default value is -1 means enagage all the processor in your computer, 4 means occupy some portion of
#the processor along with allowing other parallel tasks to be done by the processor if there are any task in your
#computer)
grid_cv.fit(x_df1,y_df1)

best_params=grid_cv.best_params_

print(best_params)

#Now let us rebuild the model with passing the best found parameters as above
dt_m2=DecisionTreeClassifier(criterion=best_params['criterion'], splitter=best_params['splitter'],
                             max_depth=best_params['max_depth'],min_samples_leaf=best_params['min_samples_leaf'],
                             min_samples_split=best_params['min_samples_split'])
dt_m2.fit(x_df1,y_df1)
print('train score cv',dt_m2.score(x_df1,y_df1))
print('test score cv', dt_m2.score(x_test,y_test))
plt.figure(figsize=(10,10))
tree.plot_tree(dt_m2,filled=True,class_names=[str(i) for i in set(y_df1)],feature_names=x_df1.columns)
plt.show()

y_predict=dt_m2.predict(x_test)
print(confusion_matrix(y_test,y_predict))














