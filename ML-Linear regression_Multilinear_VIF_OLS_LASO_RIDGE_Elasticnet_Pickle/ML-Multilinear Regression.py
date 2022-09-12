import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv('Advertising.csv')
print(df.head())
print(df.columns )

#Multi linear regression
df_X=df[['TV','radio','newspaper']]
df_y=df['sales']
print(df_X.head())
print(df_y.head())

alg_linear= LinearRegression()
alg_linear.fit(df_X,df_y)
print('intercept',alg_linear.intercept_)
print('coefficents',alg_linear.coef_)
print('score',alg_linear.score(df_X,df_y))


#Let us consider only two columns
df_X=df[['TV','radio']]
df_y=df['sales']
print(df_X.head())
print(df_y.head())

alg_linear= LinearRegression()
alg_linear.fit(df_X,df_y)
print('intercept',alg_linear.intercept_)
print('coefficents',alg_linear.coef_)
print('score',alg_linear.score(df_X,df_y))

#We can observe that there no much changin the Score (R2 square score) by removing the news paper
#column, this indicates that newspaper is not contributing much towards the increase of sales.
df_X=df[['TV','newspaper']]
df_y=df['sales']
print(df_X.head())
print(df_y.head())

alg_linear= LinearRegression()
alg_linear.fit(df_X,df_y)
print('intercept',alg_linear.intercept_)
print('coefficents',alg_linear.coef_)
print('score',alg_linear.score(df_X,df_y))

#Statistical apporch  method Modelling

import statsmodels.formula.api as smf
alg_linear_stats=smf.ols(formula='sales~TV',data=df).fit()
print(alg_linear_stats.summary())

#OLS summary of sales Vs TV and radio
import statsmodels.formula.api as smf
alg_linear_stats=smf.ols(formula='sales~TV+radio',data=df).fit()
print(alg_linear_stats.summary())

#OLS summary of sales Vs TV,radio and news paper
import statsmodels.formula.api as smf
alg_linear_stats=smf.ols(formula='sales~TV+radio+newspaper',data=df).fit()
print(alg_linear_stats.summary())



from pandas_profiling import ProfileReport as profile
fig=profile(df)
fig.to_file('profilereport.html')


#Let us work on VIF, Regularisations LASSO, Ridge regression, Elastic Net

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNetCV, LinearRegression,ElasticNet
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport as profile
df=pd.read_csv('Admission_Prediction.csv')
print(df.head())
print(df.shape)

#Requirement: We have to build a model where if i am going to give all these selected parameters(feature vaule) then the model
#should tell what is the chance of getting admission.
#
prof=profile(df)
prof.to_file('Admission_profile.html')

# Let us see which columns have missing value
print(df.isnull().sum())

#Let us see type of each columns
print(df.dtypes)

#Let us drop the Serial number
df=df.drop('Serial No.',axis=1)
print(df.head())

#Our Target label is Chance of Admit which has continous value means model must be of regression model
print(df.columns)

#Two columns  'University Rating' and 'Research' appears to be categorical variables

print(df['University Rating'].unique())
print(df['University Rating'].value_counts())
print(df['Research'].unique())
print(df['Research'].value_counts())

#Though we found 2 categorical columns, we can see that they are already having numerical values, hence we need not to handle them

#We found that there are missing values in the 3 columns

def missing_update(df,variable):
    mean_value=df[variable].mean()
    df[variable]=df[variable].fillna(mean_value)
    return df

df=missing_update(df,'GRE Score')
print(df.head())
df=missing_update(df,'TOEFL Score')
mode_val=df['University Rating'].mode()[0]
print('df[University Rating].mode()',df['University Rating'].mode()[0])
df['University Rating']=df['University Rating'].fillna(mode_val)
print(df.isnull().sum())

print(df.describe())
#Let us declare the X and y data
y=df['Chance of Admit']
X=df.drop('Chance of Admit',axis=1)
print(X.head())
print(y.head())


#Let us perform some normalization and standardization.
#Standardization is requried as we can see that each feature has ranges different from other one
#We can build the model without standardization as well but with standanrdization we can build a
#better. If the data is having various variations then the model will not be able to understand the
#relations better, if we scale down the data to a standard scale then the model would understand
#the relation between the feature in better way, that is why we usually perform standardizations.

alg_scaler=StandardScaler()
arr=alg_scaler.fit_transform(X)
X=pd.DataFrame(alg_scaler.fit_transform(X),columns=X.columns)
print(X.head())

pro_file=profile(X)
pro_file.to_file('Afternormalization.html')

print(X.describe())

#Let us check the multicolliniarity
#1 Pearson plot
#2 VIF

#VIF method this requires the np array as imput for the data
from statsmodels.stats.outliers_influence import variance_inflation_factor

print(arr.shape)
print(arr.shape[1])
#Variance inflation factors requires np array form of data with its column index

a=[variance_inflation_factor(arr,i) for i in range(arr.shape[1])]
print(a)
b=[variance_inflation_factor(arr,0)]
print(b)
#Let us store the VIF details in the data frame
vif_df=pd.DataFrame()
vif_df['VIF']=a
vif_df['Featurename']=X.columns

print(vif_df)

#VIF score is less than 10 so there is no need to handle the VIF, and we need not to drop any columns

# Let us split the data in to train and test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=52)
alg_linear=LinearRegression()
alg_linear.fit(X_train,y_train)
y_pred=alg_linear.predict(X_test)
print('Test Accuracy',alg_linear.score(X_test,y_test)) #Here we are giving unkown data i.e. test data for score
print('Train Accuracy',alg_linear.score(X_train,y_train))
print('Intercept',alg_linear.intercept_)
print('Coefficient',alg_linear.coef_)
#And remember that when you are predicting the values for th production data ensure that
#it also gone through all the preprocessing and Transmission steps otherwise the result will not be
#as expected.
#By tuning the random state value you can change your test accuracy score.
#By changing the train test split also you can change your test accuracy
#Tuning values will be of trial and error one or experimental one

#let us calculate the adjusted Rsquare
#Mehtod 1 throgh OLS method
import statsmodels.formula.api as smf
df2=X_train.copy()
df2['y_train']=y_train.copy()
print(df2.head())
print(df2.columns)
df2.columns=['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA',
       'Research','y_train']
print(df2.columns)

alg_linear_stats=smf.ols(formula='y_train~GRE_Score+TOEFL_Score+University_Rating+SOP+LOR+CGPA+Research',
                         data=df2).fit()
print(alg_linear_stats.summary())
df3=X_test.copy()
df3['y_test']=y_test.copy()
print(df3.head())
df3.columns=['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP', 'LOR', 'CGPA',
       'Research','y_test']
alg_linear_test_stats=smf.ols(formula='y_test~GRE_Score+TOEFL_Score+University_Rating+SOP+LOR+CGPA+Research',
                         data=df3).fit()
print(alg_linear_test_stats.summary())

#method 2 for finding adjusted R square
def adj_r2(x,y):
    r2=alg_linear.score(x,y)
    n=x.shape[0]
    p=x.shape[1]
    adjusted_r2=1-(((1-r2)*(n-1))/(n-p-1))
    return adjusted_r2

adj_r2_train=adj_r2(X_train,y_train)
adj_r2_test=adj_r2(X_test,y_test)
print('Adjusted r2 Train',adj_r2_train)
print('r2 Train',alg_linear.score(X_train,y_train))
print('Adjusted r2 test',adj_r2_test)
print('r2 Test',alg_linear.score(X_test,y_test))


#Let us perform Lasso Cross validation
alg_lassocv=LassoCV(cv=10,max_iter=200000,normalize=True)
#Here we can further tune the CV, max_iter values to get the better LASSO score
#CV is a cross validation technique to find out the best possible parameter by doing random experiment
#Here the entire data set will be divided into number of CV i.e 10 parts over here and it will consider
#the 9 parts for its training and 1 part for its testing purpose and it repeats this process for
#different different combination of the data set parts for training and testing, and here max_iter
#tells maximum how many such iteration it can perform (i.e. number of combinations it can iterate throgh)
#to find the best Alpha value.
#We usually use these cross validation techniques to find out the best possible hyper parameters
# which we can use it for hyper parameter tuning.
alg_lassocv.fit(X_train,y_train)
print('alg_lassocv.alpha_= ',alg_lassocv.alpha_)
#we have to first calculate the best or optimal value for lambda/alpha through LassoCV, so
#that we can use in our LASSO regression model.

alg_Lasso=Lasso(alpha=alg_lassocv.alpha_) #Here we are using th alpha value which we calculates throguh
                                          #lasso CV method
alg_Lasso.fit(X_train,y_train)
print('LASSO Score of Test', alg_Lasso.score(X_test,y_test))

#Let us perform Ridge regression
alg_ridgecv=RidgeCV(cv=10,normalize=True)
alg_ridgecv.fit(X_train,y_train)
print('alg_ridgecv alpha=',alg_ridgecv.alpha_)
# OR

alg_ridgeCV=RidgeCV(alphas=np.random.uniform(0,10,50),cv=10,normalize=True)
alg_ridgeCV.fit(X_train,y_train)
print('alg_ridgeCV alpha=',alg_ridgecv.alpha_)

alg_Ridge=Ridge(alpha=alg_ridgeCV.alpha_)
alg_Ridge.fit(X_train,y_train)
print('alg_Ridge Score',alg_Ridge.score(X_test,y_test))

#Let us perform ElasticNet
alg_elasticnetcv=ElasticNetCV(cv=10,normalize=True)
alg_elasticnetcv.fit(X_train,y_train)
print('Elastic alpha=', alg_elasticnetcv.alpha_)
print('Elastic L1 ratio=',alg_elasticnetcv.l1_ratio_)

alg_Elasticnet=ElasticNet(alpha=alg_elasticnetcv.alpha_,l1_ratio=alg_elasticnetcv.l1_ratio_)
alg_Elasticnet.fit(X_train,y_train)
print(alg_Elasticnet.score(X_test,y_test))

import pickle
Linear_file='ML_Linear_Model.pickle'
pickle.dump(alg_linear,open(Linear_file,'wb'))
Linear_unpickle=pickle.load(open(Linear_file,'rb'))
print('Linear_unpickle.score',Linear_unpickle.score(X_test,y_test))
Lasso_file='ML_Lasso_Linear_Model.pickle'
pickle.dump(alg_Lasso,open(Lasso_file,'wb'))
Ridge_file='ML_Ridge_regression_Model.pickle'
pickle.dump(alg_Ridge,open(Ridge_file,'wb'))
ElasticNet_file='ML_ElasticNet_Model.pickle'
pickle.dump(alg_Elasticnet,open(ElasticNet_file,'wb'))