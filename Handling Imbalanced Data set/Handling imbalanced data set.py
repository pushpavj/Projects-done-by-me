#first you need to install the below one
#pip install imbalanced-learn

#If the interviewer asked to handle imbalanced data set
#Answer is if the data set is small I definitely go with the under sampling but again I will be
#focusing on the all the performance metrics like precision, recall, f-1 score apart from this I
#will be focusing on the domain knowledge wheather I should be reducising the falls positive with
#the falls negative based on that I will be selecting my ROC score, i.e the probability value whether
#it should be 0.6, .7 some thing like that finally I also be doing SMOTE techniques, I will be doing
#over sampling techniques but again will be focusing on understanding the performance metrics. Finally
#if this is not working then will be going with ensemble technique like random forest, XG boost where
# i can also provide my class weight parameter so that it can perform well.

import pandas as pd
df=pd.read_csv('creditcard.csv')
print(df.head())
print(df.shape)

#let us check if there are any null values
print(df.isnull().count())
#This gives the number of records which are not null. Since all the records having total number of record
# so there are no null values in this data set.
print(df.isnull().sum())
#This gives the number of records which are actually null in each column. It is showing 0 records as null
#records for all the dolumns hence there are no null vaules in this data set.

#Let us check the dependent column out put value. We can see that column Class which is actually a
#classification problem, we need to find out from the credit card dataset that whether the credit
#card transaction is fraud or not.
#Let us find out what are the unique classifications it has
print(df['Class'].unique())
#Now let us check whether the data is balanced or not by finding how many records belogs to each class
#whether all classes have nearly equal number of records or not.
#We can say that our data is imbalanced in case of classification if the classes in the target column
#have drastic difference in the number of records for each class, if so, then it will have bias in
#in the data set for training the model and the model would be misleading
print(df['Class'].value_counts())
#We can see that class 0 has 284325 records and class 1 492, hence there is a bias in the dataset
#Let see how to solve
#Let us perform under sample
X=df.drop('Class',axis=1)
y=df['Class']
#Let us first apply ensemble technique before handling the imbalanced data set, this is because
#ensemble techqnique is known as it is not affected by the imbalanced dataset, so first let us see
#if our data works fine with this technique without handling the imbalance
from sklearn.linear_model import LogisticRegression
#Note while having imbalanced data set it is not good to check only the accuracy but we also check
#for other performance matrix so import all of them.
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
#Cross validation technique will also helps in imbalanced data set to under stand how our model is
#performing so let us import KFold for the same
from sklearn.model_selection import KFold
#So numpy required to perform KFold so let us import numpy as well
import numpy as np
#Also let us perform gridsearch CV so we do not miss any thing
from sklearn.model_selection import GridSearchCV
#Later we also check with ensemble technique
alg_logistic=LogisticRegression()
#Logisticregression has certain hyper parameter, so le us use couple of them to see if my algotithm
#works better with the imbalance data set
grid={'C':10.0**np.arange(-2,3),'penalty':['l1','l2']}
#also use the kfold
cv=KFold(n_splits=5,shuffle=False,random_state=None)
#You can have different different scoring parameter do ctrl+Q to see what are all the scores available
clf=GridSearchCV(alg_logistic,grid,cv=cv, n_jobs=-1,scoring='f1_macro')
#Before applying the algorithm on the data let us do train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,train_size=0.7)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print('Confusion_matrix')
print(confusion_matrix(y_test,y_pred))
print('Accuracy Score',accuracy_score(y_test,y_pred))
#Here it shows 99% of accuracy, for imbalanced data set if you are getting better accuracy then do
#thing that is good, actually i.e. bad. You have to look at other features like prcision, recall
# other metrix as well
#Here actually the accuracy is more because there are more zero records compared to 1, so most of the
#time it had been to predict 0 and in case of predicting test scinarios also the number of records to be
#be predicted were of 0's, hence it was able to predict the test data more accurately. Since the total
#number of recods to be predicted were most of 0's only, which covers more % of the test data
# hence it was able predict 0's most of the time, hence it was covering large % of accuracy.
print('Classification Report',classification_report(y_test,y_pred))
#Here we can see that precission and recall for predicting 0 is 1, this indicates the model is able
#to predict 0's nearly 100% correct, this has hapenned because of model is trained with more number of
#0 records, however the precission recall for predicting 1 is also between 69 to 74% which is not
# bad, though the data set was imballanced for 1 records, this hapenned due to hyper parameter tuning
# and cross validations we applied during the model building.


#Now let us build the Ensemble module with Randforestclassifier. Randomforest is the ensemble techniqu
#which has many discision trees. Discision tree plays an important role, basically it forms the
#hierarchical structure with an imbalanced data set which gives the very good accuracy
from sklearn.ensemble import RandomForestClassifier
alg_rand=RandomForestClassifier()
alg_rand.fit(X_train,y_train)
y_pred=alg_rand.predict(X_test)
print('Confusion matrix')
print(confusion_matrix(y_test,y_pred))
print('accuracy',accuracy_score(y_test,y_pred))
print('classification report',classification_report(y_test,y_pred))
#Here we can observe that initially with LOgistic regression there were around 46 falls positives
#now it is reduced to 5 and increated the value of TP, hence we can say that the descion tree
#do not have much impact with the imbalanced data sets, it works pritty much well even with the
# the imbalanced data set. Here we can see the precision score for 1 as 95% and F1-score as 95 with
#recall of 77% which is very good. This is with out applying cross validation and hyper parameter tuning
#If you apply those as well this algorithm works even better.

#Under sampling: This is to reduce the points or weightage of maximum lables (i.e. the class which
# has maximum number of records, ). This method also has some disadvantage becuasue it leads to
#some loss of data, hence this method is not used most of the time. You can probabily use this method
#when there is very less data set itself. Let us see how to perform
from imblearn.under_sampling import NearMiss
print(y_train.value_counts())
ns_alg=NearMiss(.80) #reduce  0 class parameters numbers to a number which gives
# 80% of the new number gives total number of 1 class records.
#i.e. in this case it will take count of records for 0 as number n such that 80% of n should give
#actual total number of records of class 1.
#here, actual total number of records of class 1 is 356, so Nearmiss found the number n as 445 such
#that 80% of 445 giving 356.
X_train_ns,y_train_ns=ns_alg.fit_resample(X_train,y_train)
from collections import Counter
print(f'The number of class before the fit {Counter(y_train)}')
print(f'The number of class after the fit {Counter(y_train_ns)}')

#Now let us perform Randomforest using this data,
from sklearn.ensemble import RandomForestClassifier
alg_rand=RandomForestClassifier()
alg_rand.fit(X_train_ns,y_train_ns)
y_pred=alg_rand.predict(X_test)
print('Confusion matrix')
print(confusion_matrix(y_test,y_pred))
print('accuracy',accuracy_score(y_test,y_pred))
print('classification report',classification_report(y_test,y_pred))
#Here we can observe that with the under sampling the Random forest did not perform well as
#the number of training records were very less due to under sampling

#Oversampling
from imblearn.over_sampling import RandomOverSampler
alg_os=RandomOverSampler(.50) # in this case number of records for class 1 will get increased up to
#50% of actual number of class 0. It auto matically creates the sample records for class 1 based on
#actual class 1 records.
X_train_os,y_train_os=alg_os.fit_resample(X_train,y_train)
print(f'The number of class before the fit {Counter(y_train)}')
print(f'The number of class after the fit {Counter(y_train_os)}')


#Now let us perform Randomforest using this data,
from sklearn.ensemble import RandomForestClassifier
alg_rand=RandomForestClassifier()
alg_rand.fit(X_train_os,y_train_os)
y_pred=alg_rand.predict(X_test)
print('Confusion matrix')
print(confusion_matrix(y_test,y_pred))
print('accuracy',accuracy_score(y_test,y_pred))
print('classification report',classification_report(y_test,y_pred))
#We can observe that with the oversampling it reduced the falls negatives which is very good. Even
#further if we apply performanc tuning we can have more good results.
#Let us try with 75% of oversampling as see
from imblearn.over_sampling import RandomOverSampler
alg_os=RandomOverSampler(.75) # in this case number of records for class 1 will get increased up to
#50% of actual number of class 0. It auto matically creates the sample records for class 1 based on
#actual class 1 records.
X_train_os,y_train_os=alg_os.fit_resample(X_train,y_train)
print(f'The number of class before the fit {Counter(y_train)}')
print(f'The number of class after the fit {Counter(y_train_os)}')

from sklearn.ensemble import RandomForestClassifier
alg_rand=RandomForestClassifier()
alg_rand.fit(X_train_os,y_train_os)
y_pred=alg_rand.predict(X_test)
print('Confusion matrix')
print(confusion_matrix(y_test,y_pred))
print('accuracy',accuracy_score(y_test,y_pred))
print('classification report',classification_report(y_test,y_pred))

#SMOTETomek technique
#It uses the combination of undersampling and oversampling over here. Here it will create the
#new near by records as additional records for lower class (here class 1). Compared to under sampling
#and oversampling it takes more time for execution. In case of over sampling it creates the additional
#points same as actual existing lower class but here it creates near by records it creates.
#In case of SMOTE internally it uses K nearest neighbout to create the new point
from imblearn.combine import SMOTETomek
alg_smote=SMOTETomek()
X_train_smote,y_train_smote=alg_smote.fit_resample(X_train,y_train)
print(f'The number of class before the fit {Counter(y_train)}')
print(f'The number of class after the fit {Counter(y_train_smote)}')
from sklearn.ensemble import RandomForestClassifier
alg_rand=RandomForestClassifier()
alg_rand.fit(X_train_smote,y_train_smote)
y_pred=alg_rand.predict(X_test)
print('Confusion matrix')
print(confusion_matrix(y_test,y_pred))
print('accuracy',accuracy_score(y_test,y_pred))
print('classification report',classification_report(y_test,y_pred))

#Ensemble technic under imblearn library
from imblearn.ensemble import EasyEnsembleClassifier
alg_easy=EasyEnsembleClassifier()
alg_easy.fit(X_train,y_train)
y_pred=alg_easy.predict(X_test)
print('Confusion matrix')
print(confusion_matrix(y_test,y_pred))
print('accuracy',accuracy_score(y_test,y_pred))
print('classification report',classification_report(y_test,y_pred))
#It is giving bad result, you need to apply some tuning parameter over here so that it give good
#resulst.
#Evern there is a isolation technique present to handle the imbalanced data set
