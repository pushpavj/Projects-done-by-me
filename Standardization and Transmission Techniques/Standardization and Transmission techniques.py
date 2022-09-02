#Why transformation of feature are required
#In ML algorithm, we have global minima and we always look for finding the global minima as quickly as
#possible, and incase of K nearest neighbour algorithem we look for Uqlidian distance. In case of
#K means also we look for uqlidian distance. If we have multiple features with large difference between
#them, i.e, one feature has value as 10, 20, 33,44,55 and antoher feature has value as 700, 850,340...etc
#in such cases when the difference is more then finding the global minima or uqlidan distance will be time
#taking or will be difficult, if the difference is minimum then it will be quick and smooth to find the
#global minima and Uqlidian distance.
#Every point has vector and direction which will have influence on finding the global minima and uqlidian
#distance.
#Is every algorithm requires a standardization and transmission of the data?
#Anwer is no we need do it for algorithm such as linear regression,logistic regression, K nearest neighbour
#K means,heirarchical means etc... where gradiant discent concep is used, or uqlidian distance is
# used or where you need to find out the global minima is such cases we need to perform standardization
# and transmission of the data We do not require transformation in algorithm such as decision tree,
# random forest, XG boost, ensemble techniques.
# Deep learning techniques ANN, CNN, RNN in all these techniques you require standardization and
# transmission.
#In this we have following technique
#Normalisation and Standardization
#Scaling to minimum and Maximum values
#Scaling to Median and Quantiles
#Guassiain trasformations as follows
    # Logerthemic transformation
    # Reciprocal transformation
    # Square root transformation
    # Exponenetial transformation
    # Box COX transformations

#Standardization
#In this technique we will bring all the features to similar scale
#Standardization means centring the variable at zero
#z=(x-x_mean)/std
#We use the standard scaler from sklearn library

#What is the difference between fit and fit transform
# W r t ML algorithm we only do the fit , we train our data with that
#And suppose if you want to transform or change the data at that time we use fittransform
#i.e apply the algorithm and then also change the data, then we use fit transformation
#here data transformation will happen. The transformation will happen feature or column vise
#not by row wise

#if there are any outliers in the data then there will be some impact on the transformations


import pandas as pd
df=pd.read_csv('titanic.csv',usecols=['Pclass','Age','Fare','Survived'])

print(df.head())
#Check if any null values
print(df.isnull().sum())
#There are null values in Age column so let us replace them with median
df['Age']=df['Age'].fillna(df['Age'].median())
print(df.isnull().sum())
#We use the standard scaler from sklearn library
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df)
print(df_scaled)
cols=df.columns
df_transformed=pd.DataFrame(df_scaled,columns=cols)
print(df_transformed)

import matplotlib.pyplot as plt
#%matplotlib inline
plt.hist(df_scaled[:,1],bins=20)
plt.show()
plt.hist(df_transformed['Survived'],bins=20)
plt.title('Survived')
plt.show()
plt.hist(df_transformed['Pclass'],bins=20)
plt.title('Pclass')
plt.show()
plt.hist(df_transformed['Age'],bins=20)
plt.title('Age')
plt.show()
plt.hist(df_transformed['Fare'],bins=20)
plt.title('Fare')
plt.show()

#Min Max scaling - it is popularly used in Deeplearning techniques such as CNN
#Min max scaleing transforms the value between 0 and 1
#Formula= X-X_min/X_max-X_min
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
df_minmax=pd.DataFrame(min_max.fit_transform(df),columns=cols)
print(df_minmax.head())

plt.hist(df_minmax['Survived'],bins=20)
plt.title('Survived')
plt.show()
plt.hist(df_minmax['Pclass'],bins=20)
plt.title('Pclass')
plt.show()
plt.hist(df_minmax['Age'],bins=20)
plt.title('Age')
plt.show()
plt.hist(df_minmax['Fare'],bins=20)
plt.title('Fare')
plt.show()

#
# #Robost Scalar
# #It is used to scale the features to median and Quantiles
# #Formula X_Scaled=(X-X_median)/IQR
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df_robust_scaler=pd.DataFrame(scaler.fit_transform(df),columns=cols)
print(df_robust_scaler.head())
plt.hist(df_robust_scaler['Survived'],bins=20)
plt.title('Survived')
plt.show()
plt.hist(df_robust_scaler['Pclass'],bins=20)
plt.title('Pclass')
plt.show()
plt.hist(df_robust_scaler['Age'],bins=20)
plt.title('Age')
plt.show()
plt.hist(df_robust_scaler['Fare'],bins=20)
plt.title('Fare')
plt.show()

#Guassian Distribution
#What is Guassina Distribution, when we have data which is not normally distributed then we apply some
#mathematical rules to it to convert them to Guassian Distributions.
#Some of the algorrithms such as linear regressions, logistic regression works very well when they are
#Guassian distrubuted.
df=pd.read_csv('titanic.csv',usecols=['Age','Fare','Survived'])
print(df.head())
df['Age']=df['Age'].fillna(df['Age'].median())
print(df.head())
import scipy.stats as stat
import pylab
#If you want to check whether the data is guassian distributed or not we use QQ plot
def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist='norm',plot=pylab)
    plt.show()
plot_data(df,'Age')
#we can observe that Most of the data in Age in fall on the line, i.e. it is nearly normally distrbuted
#Let's do logorthmic transmission and see the distribution
import numpy as np
df['Age_log']=np.log(df['Age'])
plot_data(df,'Age_log')
# #We can observe that with the log transmission the most of the data fall out of the line hence we can
# #say that the log transmission is not suitable for this Age feature
#
plot_data(df,'Fare')
for i in df['Fare']:
    if i==0:
        print('Fare has zerovalues in it')
#Since Fare has zero in it log transimssion does not work for it.
#
# df['Fare_log']=np.log(df['Fare'])
# plot_data(df,'Fare_log')

#Reciprocal transformations
df['Age_reverse']=1/df['Age']
plot_data(df,'Age_reverse')

#Square root transmission
df['Age_square']=df['Age']**(1/2)

plot_data(df,'Age_square')

#Square root transmission on Age feature appearse to be good

#Exponential transformation
df['Age_expo']=df['Age']**(1/1.2)
plot_data(df,'Age_expo')
# #Exponential transmission on the Age feature appears pritty much good compared to other transmissions
df['Age_expo']=df['Age']**(1/1.32)
plot_data(df,'Age_expo')

#BOXCOX transformation
#Formula T(Y)=(y expo(lamda)-1)/lamda, Lamda varies between -5 to +5
df['Age_boxcox'],parameter=stat.boxcox(df['Age'])
#Here the parameter gives the value of lambda used which will be between -5 to +5
print(parameter)

plot_data(df,'Age_boxcox')

#The boxcox transformation has done pritty well compared to exponential for Age feature.
#Out of all transoframtion for AGE we can go without transformation as the data is almost
#normally distributed or we can select either Boxcox or exponenetial transformations.

#Since Fare was having zero values log transformation was not working so we have log1p
#now we can use log1p instead of log. Actually the log1p will add value 1 to the Fare before
#applying log on it.
df['Fare_log']=np.log1p(df['Fare'])
plot_data(df,'Fare_log')

#Instead of log1p even we can use log as below by adding 1 manually to the fare
df['Fare_log']=np.log(df['Fare']+1)
plot_data(df,'Fare_log')
#Log transformation worked very well for Fare, it made most of the data to fall on the line
#Let us now try with BOXCOX transformation, here also we are going to add 1 to fare as
#it is not working as it was forming negative value during boxcox transformation.
df['Fare_boxcox'],parameter=stat.boxcox(df['Fare']+1)
plot_data(df,'Fare_boxcox')
#compared to log transform and boxcox, the log transform is working well for Fare
