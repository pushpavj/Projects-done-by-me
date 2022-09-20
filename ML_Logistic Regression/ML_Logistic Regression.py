import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV,Lasso,Ridge,ElasticNet
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,auc,roc_auc_score
import seaborn as sns
import pickle

df=pd.read_csv(r'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')
print(df.head())
profile=ProfileReport(df)
profile.to_file('Diabetes_Profile.html')
#Let us print the number of zeroes present in each columns, we can get these details from profile also
print(df.isin([0]).sum())
print('Null values')
print(df.isnull().sum())
print(df.isna().sum())
print(df.isin(['nan']))
#Let us handle the columns with zeroes
print(df.columns)
print(df['BMI']==0)
print(df['BMI'] [df['BMI']==0])
print(df['BMI'] [df['BMI']==0].count())
df['BMI'].replace(0,df['BMI'].mean(),inplace=True)
print(df['BMI'] [df['BMI']==0].count())
df['BloodPressure'].replace(0,df['BloodPressure'].mean(),inplace=True)
df['SkinThickness'].replace(0,df['SkinThickness'].mean(),inplace=True)
df['Insulin'].replace(0,df['Insulin'].mean(),inplace=True)
df['Glucose'].replace(0,df['Glucose'].mean(),inplace=True)
#Let us print the number of zeroes present in each columns
print(df.isin([0]).sum())
#Pregnancies columns having zeroes is acceptable

#Let us see the distribution of the dat throgh profile once again
# profile=ProfileReport(df)
# profile.to_file('Afterzero.html')

#Let us find and handle the out lier
#If there is a skewness in the data set means there is outlier
#Let us do box plot

import plotly.express as pltl
fig=pltl.box(df)
fig.show()

#Insulin column has large number of outliers Outliers are present in BloodPressure, SkinThickness
#BMI,Age,

#Let us treat these out liers
#Let us apply the quantile method which will reduce the number of observations
#It will keep the % of data in between the interquantile range and it will remove the data above and
#below of it.
#We can apply this method using each individual column quantile and see if the outliers are reduced or
#not. We can tune the % accordingly. However by treating one column with this method will also
#affect the rest of the columns as for each treatment it will remover some number of observations
#due to which rest of the columns will also be affected.
#However through trial and error method I found that applying this method to whole dataframe
# itself(not individually) has reduced lot many outliers in most of the column.
# With keeping 97% of data outliers of Pregnancies is removed, however there is some data loss in this
# method
q=df['Pregnancies'].quantile(.98)
df_new=df[df['Pregnancies']<q]
print(df_new)

q=df['BMI'].quantile(.99)
df_new=df[df['BMI']<q]
print(df_new)


q=df['SkinThickness'].quantile(.99)
df_new=df[df['SkinThickness']<q]
print(df_new)

q=df['Insulin'].quantile(.99)
df_new=df[df['Insulin']<q]
print(df_new)
#However through trial and error method I found that applying this method to whole dataframe
# itself(not individually) has reduced lot many outliers in most of the column but this also
#made the outcome column which is a target column to have observations with only one category
#which is not accepted hence commented out
# q=df.quantile(.98)
# df_new=df[df<q]
# print(df_new)
# print('Null in df_new')
# print(df_new.isnull().sum())
# q=df['SkinThickness'].quantile(.99)
# df_new=df[df['SkinThickness']<q]
# print(df_new)
#df_new=df.copy()
df_new.reset_index(inplace=True)
df_new.drop(['index'],axis=1,inplace=True)
import plotly.express as pltl
fig=pltl.box(df_new)
fig.show()
# profile=ProfileReport(df_new)
# profile.to_file('After_Quantile_Profile.html')

#If we lookinto the profiles after quantile treatment and compared with its previous profile we can
#found that now the data is getting quite normally distributed.
#Through quantile we were not able to remove all the outliers in all the columns, some columns are
#still need to be treated further for removal of outliers
a=df['Insulin'].quantile(.75)
def outlier_removal(df,variable):
    q1=df[variable].quantile(.25)
    q3=df[variable].quantile(.75)
    IQR=q3-q1
    maxlimit=q3+1.5*IQR
    minlimit=q1-1.5*IQR
    index_list=[]
#    index_list=[i for i in range(len(df[variable])) if df[variable][i] <=maxlimit]
#    index_list = [i for i in range(len(df[variable])) if df[variable][i] >= minlimit and df[variable][i] <= maxlimit]
    for i in range(len(df[variable])):
        if df[variable][i] >= minlimit and df[variable][i]<=maxlimit:
      #      print(df[variable][i])
            index_list.append(i)
    df_new1=df.loc[index_list,df.columns]
    print(df_new1.head())
    print(df_new1.columns)
    df_new1.reset_index(inplace=True)
    df_new1.drop('index',axis=1,inplace=True)
    return df_new1
#df_new=outlier_removal(df_new,'Insulin')
df3=outlier_removal(df_new,'BloodPressure')

df4=outlier_removal(df3,'Insulin')
df4=outlier_removal(df4,'Age')
df4=outlier_removal(df4,'SkinThickness')
df4=outlier_removal(df4,'BMI')
df4=outlier_removal(df4,'BloodPressure')
df4=outlier_removal(df4,'DiabetesPedigreeFunction')
df4=outlier_removal(df4,'Glucose')
#df4=outlier_removal(df4,'Insulin')
#df4=outlier_removal(df4,'Insulin')
fig=pltl.box(df4,title='Insulin outlier removal')
fig.show()
print(df_new.head())
#When we are removing the outlier we must remember that with the removal of outliers we are also
#removing observations related to the outliers which will cuase the data loss as well. we must be
#careful, some time we will use the data with outliers in it or in some time we will take out all
#the observations related to the outliers and build the seperate model for the same,
#for example Insulin column is having a large number of out liers, in that case we can seperate
#these observations into a seperated data frame and build the seperate model for the same.
#And during the predicting for the new data we can check for the Insulin column value for the
#maxlimit and minlimit and if it falls within this limit it will be passed theough the one algorithm
#for prediction else it will be passed through the second model (outlier one) for prediction.
#Since when we are treating the outlier for one column and due to data losss it will affect the
#other columns as well, so you may find that reducing the outlier in one column may increase the
#outlier in another column as well, so it is always not possible to remove the outliers in all the
#columns.
# profile=ProfileReport(df4)
# profile.to_file('AfterIQR.html')

#Let us apply log transmission
import numpy as np
#df4['Pregnancies_log']=np.log(df4['Pregnancies'])
#df4['Age_log']=np.log(df4['Age'])

#Let us split the data now
X=df4.drop(['Outcome'],axis=1)
y=df4['Outcome']

#Let us perform standardization as the data is fluctuating a lot
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
print(X_scaled)
cols=X.columns
#_transformed=pd.DataFrame(X_scaled,columns=cols)
X_transformed=pd.DataFrame(X_scaled)
print('X_transformed\n',X_transformed)
# profile=ProfileReport(X_transformed)
# profile.to_file('Afterstandard.html')

fig=pltl.box(X_transformed,title='Standardized data')
fig.show()

#Now let us find out the VIF for identifying multicolliniarity

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
#
#Variance inflation factors requires np array form of data with its column index
#X_scaled is in array form
a=[VIF(X_scaled,i) for i in range(X_scaled.shape[1])]

#Let us store the VIF details in the data frame
vif_df=pd.DataFrame()
vif_df['VIF']=a
vif_df['Featurename']=X.columns

print('VIF Values are \n', vif_df)
#If we look in to the VIF score it is less than 10 for all the features so we can say that
#we need not to handle the multicolliniarity in this case

#Note we have applied standard scalar before the train test split here, this is because
#the standard scaler just scales the data to different scales there is no chance of data
#lickage over here, hence we can perform the standard scaler transmission before the train and
#test split
#Let us do train and Test split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,train_size=.20,random_state=144)

#print(X_train.head())
print('X_train[0]\n',X_train[0])

alg_logistic=LogisticRegression(verbose=1)
#Following are the solver available under this LogisticRegression available
#Solver is nothing but a different different method or appraoch to perform a classification
#you can do ctrl+Q for more details. We can choose any of the below options.
#Each solver uses its own regularizations
#L-BFGS-B – Software for Large-scale Bound-constrained Optimization
# LIBLINEAR – A Library for Large Linear Classification and only for binary classification it can
#handle
# SAG – Minimizing Finite Sums with the Stochastic Average Gradient
# SAGA –  A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite
# Objectives"
# Hsiang-Dual coordinate descent methods for logistic regression and maximum entropy models.
#Newton CG- In this elastic regularization is present

#The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization with primal
# formulation, or no regularization. The 'liblinear' solver supports both L1 and L2 regularization,
# with a dual formulation only for the L2 penalty. The Elastic-Net regularization is only supported
# by the 'saga' solver.

#Based on our problem statement, size and type of classifications we can use different different
#methods mentioned above can be used.

#In case of multi class classification what is the lost function is used. Answer Categorical
#cross entropy loss

#We need to perform different different experiments with the different different aproaches metioned
#inside the Logistic regression algorithm. The knowledge will come on which approach suits for
#what kind of data and problem statement can come only through practice. You need to do practice
#more on all these 10 20 algorithms with different different regularization factors and all
#available parameters.Then only you will be able to answer in the interview.
#search in google for sklear.LogisticRegression and look in to the sklearn library for detailed info
#Verbose is a parameter, it is kind of logging i.e gives the logs if the jobs failuer like that
#n_jobs parameter will allow you to control the percentage of CPU allocation for your program
#

#Note here X_train and X_test are in array format or in list as it is taken from X_scaled
print('X_test[0]\n',X_test[436])
alg_logistic.fit(X_train,y_train)
y_predict=alg_logistic.predict_proba([X_test[0]]) #Gives the probabiliteis of each class 0 and
#class 1. Since X_test[0] is a array list so making it in 2 dimentional array
print(y_predict)
y_predict=alg_logistic.predict([X_test[0]]) #Gives the prediction of the given data. i.e. prdeicts
#the class based on the class which has heighest probability. Here X_test[0] is array list hence
#using [] making it as 2 dimentional array.
print(y_predict) #here it is wrongly predicting the class for X_test[0] as 0
print('y[0] actual ', y[0]) #Actual prediction of X_test[0] is 1

y_log_probability=alg_logistic.predict_log_proba([X_test[0]])
print(y_log_probability)

#Let us use the Liblinear solver
alg_logistic_liblinear=LogisticRegression(solver='liblinear',verbose=1)
alg_logistic_liblinear.fit(X_train,y_train)
y_liblinear_predict=alg_logistic_liblinear.predict(X_test)
y_logistic_predict=alg_logistic.predict(X_test)

print(confusion_matrix(y_test,y_logistic_predict))
print(confusion_matrix(y_test,y_liblinear_predict))

def model_eval(y_actual,y_pred):
    tn,fp,fn,tp=confusion_matrix(y_actual,y_pred).ravel()
    print('tn:',tn,'fp:',fp,'fn:',fn,'tp:',tp)
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    specificity=tn/(tn+fp)
    f1_score=2*(precision*recall)/(precision+recall)
    roc_aucscore=roc_auc_score(y_actual,y_pred)
    fpr,tpr,threshold=roc_curve(y_actual,y_pred)
    result={'Accuracy':accuracy,'precision':precision,'Recall':recall,'Specificity':specificity,
            '\n':'\n',
            'F1_score':f1_score,'AUC':roc_aucscore,'FPR':fpr,'TPR':tpr,
                                                                          'Threshod':threshold}

    return result
result_1_logistic=model_eval(y_test,y_logistic_predict)
result_2_Liblinear=model_eval(y_test,y_liblinear_predict)
print(result_1_logistic)
print(model_eval(y_test,y_liblinear_predict))

#Let us build the ROC Curve Plot

plt.plot(result_1_logistic['FPR'],result_1_logistic['TPR'],color='orange',label='ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='ROC Curve (area=%0.2f)'%result_1_logistic['AUC'])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic (ROC) Curve')
plt.legend()
plt.show()


plt.plot(result_2_Liblinear['FPR'],result_2_Liblinear['TPR'],color='orange',label='ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='ROC Curve (area=%0.2f)'%result_2_Liblinear['AUC'])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic (ROC) Curve Lib Linear solver')
plt.legend()
plt.show()



