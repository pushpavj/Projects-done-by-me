import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly import express as pltl

df=pd.read_csv('titanic.csv')

#Distributive plot: From the distributive plot we can check the skewness to identify
# the out liers. In the below plot we can identify that the plot is slightly rightly
# skewed.

sns.distplot(df['Age'].dropna())
plt.show()

sns.distplot(df["Age"])
plt.show()

fig=pltl.histogram(df['Age'])
fig.write_html('hist.html',auto_open=True)

#Boxplot: From the Box plot we can identify the outliers as the values which are
# above upper fence and below lower fence.

fig=pltl.box(df['Age'])
fig.write_html('Age_box.html',auto_open=True)
#
#Scatter plot: The below scatter plot is plotted between the Age column value
# against it’s index number. From the scatter plot we can observe that there are
# only few counts of Age above 70. I.e. out of around 900 there are only 5 to 6 of
# them are of age above 70 which can be identified as outliers.

fig=pltl.scatter(df['Age'])
fig.write_html('scatter.html',auto_open=True)
print(df["Age"].describe())

#Before Handling outliers we must handle the missing values in it
df_mean=df['Age'].mean()
df_std=df['Age'].std()
df2=df
df2['Age']=df['Age'].fillna(df_mean)
print(df2.head())
# Handling the out liers with z-score ore extreme dimentions or above or below 3
# standard diveiation
df2_Age_mean=df2['Age'].mean()
df2_Age_std=df2['Age'].std()
up_limit=df2_Age_mean+df2_Age_std*3
low_limit=df2_Age_mean-df2_Age_std*3

outlier_list=[i for i in df2['Age'] if i<low_limit or i > up_limit]
print('outlier_list',outlier_list)
print(df.columns)
index_list=[i for i in range(len(df2['Age'])) if df2['Age'][i]>=low_limit and df2['Age'][i]<=up_limit]
df_new=df2.loc[index_list]
print(df_new.head())
print(df_new.shape)
#df_new
outlier_list2=[i for i in df_new['Age'] if i<low_limit or i > up_limit]
print(outlier_list2)
print(df2.shape)

sns.distplot(df_new['Age'])
plt.show()

fig=pltl.scatter(df_new['Age'])
fig.write_html('scatter_extreme.html',auto_open=True)
fig=pltl.box(df_new['Age'])
fig.write_html('boxplot_extreme.html',auto_open=True)

# # Another way of handling outlier is with IQR (inter quartile range)
fig=pltl.box(df2['Age'])
fig.write_html('box_df2.html',auto_open=True)
df2_Age_q1=df2['Age'].quantile(.25)
df2_Age_q3=df['Age'].quantile(.75)
df2_Age_IQR=df2_Age_q3-df2_Age_q1
df2_Age_maxfence=df2_Age_q3+1.5*df2_Age_IQR
df2_Age_minfence=df2_Age_q1-1.5*df2_Age_IQR
#
#
outlier_list=[i for i in df2['Age'] if i< df2_Age_minfence or i> df2_Age_maxfence]
print('outlier_List', outlier_list)
print('Lenght of outlier',len(outlier_list))
index_list=[i for i in range(len(df2['Age'])) if df2['Age'][i] >=df2_Age_minfence and df2['Age'][i] <= df2_Age_maxfence]

print('index_list',index_list)
print('lenght_index',len(index_list))
df_new=df2.loc[index_list,df2.columns]
print('df_new',df_new.head())
print('df_new_shape',df_new.shape)



fig=pltl.scatter(df_new['Age'])
fig.write_html('scatter_IQR.html',auto_open=True)

fig=pltl.box(df_new["Age"])
fig.write_html('box_IQR.html',auto_open=True)
sns.distplot(df_new['Age'])
plt.show()


# #Another way of identifying the outlier is using z-score value
#
df2_Age_mean=df2['Age'].mean()
df2_Age_std=df2['Age'].std()
outlier_list=[]
outlier_index=[]
upper_threshold_std=3
lower_threshold_std=-3
for i in range(len(df2['Age'])):
    z=(df2['Age'][i]-df2_Age_mean)/df2_Age_std
    if z > upper_threshold_std or z< lower_threshold_std:
        outlier_list.append(df2['Age'][i])
    if z <= upper_threshold_std and z>= lower_threshold_std:
        outlier_index.append(i)

print('outlier_list',outlier_list)
print('Length_outlier',len(outlier_list))
print('Length_index',len(outlier_index))

df_new=df2.loc[outlier_index,df2.columns]
print("df_new.head()", df_new.head())
print("df_new shape",df_new.shape)

sns.distplot(df_new['Age'])
plt.show()

fig=pltl.box(df_new['Age'])
fig.write_html('Box_zscore.html',auto_open=True)

fig=pltl.scatter(df_new['Age'])
fig.write_html('Scatter_zscore.html',auto_open=True)

#Another way of Handling outlier is using DBSCAN
from sklearn.cluster import DBSCAN

#Tune eps and min_samples value till you get desire pattern when you plot the
#scatter plot
alg_dbscan=DBSCAN(eps=2,min_samples=10)
model=alg_dbscan.fit(df2[['Age']])
print('model',model)
colors=model.labels_


fig=pltl.scatter(df2[['Age']],color=colors)
#plt.show()
fig.write_html('dbscan.html',auto_open=True)

plt.scatter(df2['Age'],df2.index, c=colors)
plt.show()

df2['Age_labels']=colors
print(df2["Age_labels"].head())
print(df2["Age_labels"].unique())
print(df2["Age_labels"].value_counts())
#
outlier_list=[i for i in df2['Age_labels'] if i==-1]
non_outlier_index=[i for i in range(len(df2['Age_labels'])) if
df2['Age_labels'][i]==0]
df_new=df2.loc[non_outlier_index,df2.columns]
print(df_new.head())

sns.distplot(df_new['Age'])
plt.show()

fig=pltl.box(df_new['Age'])
fig.write_html('DBSCAN_box.html',auto_open=True)

fig=pltl.scatter(df_new['Age'])
fig.write_html('DBSCAN_Scatter.html',auto_open=True)
#
# #Anonter way of listing Outlier using dbscan model directly is
outlier_list2=df2[model.labels_==-1]
print('outlier_list2',outlier_list2)

#some trials
#Tune eps and min_samples value till you get desire pattern when you plot the
# scatter plot
alg_dbscan=DBSCAN(eps=2,min_samples=35)
model=alg_dbscan.fit(df2[['Age']])
print('model',model)
colors=model.labels_


fig=pltl.scatter(df2[['Age']],color=colors)
#plt.show()
fig.write_html('dbscan.html',auto_open=True)

plt.scatter(df2['Age'],df2.index, c=colors)
plt.show()
df2['Age_labels']=colors
print(df2["Age_labels"].head())
print(df2["Age_labels"].unique())
print(df2["Age_labels"].value_counts())

outlier_list=[i for i in df2['Age_labels'] if i==-1]
non_outlier_index=[i for i in range(len(df2['Age_labels'])) if
df2['Age_labels'][i]==0 or df2['Age_labels'][i]==1]
df_new=df2.loc[non_outlier_index,df2.columns]
print(df_new.head())

sns.distplot(df_new['Age'])
plt.show()

fig=pltl.box(df_new['Age'])
fig.write_html('DBSCAN_box.html',auto_open=True)

fig=pltl.scatter(df_new['Age'])
fig.write_html('DBSCAN_Scatter.html',auto_open=True)

#some trials2

#some trials
#Tune eps and min_samples value till you get desire pattern when you plot the
# scatter plot
alg_dbscan=DBSCAN(eps=2,min_samples=35)
model=alg_dbscan.fit(df2[['Age']])
print('model',model)
colors=model.labels_


fig=pltl.scatter(df2[['Age']],color=colors)
#plt.show()
fig.write_html('dbscan.html',auto_open=True)

plt.scatter(df2['Age'],df2.index, c=colors)
plt.show()
df2['Age_labels']=colors
print(df2["Age_labels"].head())
print(df2["Age_labels"].unique())
print(df2["Age_labels"].value_counts())

outlier_list=[i for i in df2['Age_labels'] if i==-1]
non_outlier_index=[i for i in range(len(df2['Age_labels'])) if
df2['Age_labels'][i]==0]
df_new=df2.loc[non_outlier_index,df2.columns]
print(df_new.head())

sns.distplot(df_new['Age'])
plt.show()

fig=pltl.box(df_new['Age'])
fig.write_html('DBSCAN_box.html',auto_open=True)

fig=pltl.scatter(df_new['Age'])
fig.write_html('DBSCAN_Scatter.html',auto_open=True)


#Tune eps and min_samples value till you get desire pattern when you plot the
# scatter plot
alg_dbscan=DBSCAN(eps=.5,min_samples=20)
model=alg_dbscan.fit(df2[['Age']])
print('model',model)
colors=model.labels_


fig=pltl.scatter(df2[['Age']],color=colors)
#plt.show()
fig.write_html('dbscan2.html',auto_open=True)

plt.scatter(df2['Age'],df2.index, c=colors)
plt.show()


#
# LOF local outlier Factor
# using Local Outlier Factor (LOF)
#
# The anomaly score of each sample is called Local Outlier Factor. It measures
# the local deviation of density of a given sample with respect to its neighbors.
# The anomaly score depends on how isolated the object is with respect to the
# surrounding neighborhood. More precisely, locality is given by k-nearest
# neighbors, whose distance is used to estimate the local density. By comparing
# the local density of a sample to the local densities of its neighbors, one
# can identify samples that have a substantially lower density than their
# neighbors. These are considered outliers.
from sklearn.neighbors import LocalOutlierFactor
alg_lof=LocalOutlierFactor(n_neighbors=10,p=2)
# # use fit_predict to compute the predicted labels of the training samples
# # (when LOF is used for outlier detection, the estimator has no predict,
# # decision_function and score_samples methods).
model=alg_lof.fit_predict(df2[['Age']])
print(model)
non_outlier_list=model != -1
outlier_list=model ==-1
print('Length of outlier',sum(outlier_list))
print(non_outlier_list)
print('outlier_list',outlier_list)
print(df2.loc[non_outlier_list,'Age'])
print(df2.loc[outlier_list,'Age'])
print(df2.loc[outlier_list,'Age'].unique())
df_new=df2.loc[non_outlier_list,df2.columns]
print(df_new.head())
print(alg_lof.negative_outlier_factor_)
df2['Age_LoF']=alg_lof.negative_outlier_factor_.tolist()

fig=pltl.scatter(df2[['Age']],color=model)
fig.write_html('lof_scatter.html',auto_open=True)
# fig=pltl.scatter(df_new[['Age']],color=model)
# fig.write_html('lofscatter.html',auto_open=True)
sns.distplot(df_new['Age'])
plt.show()

fig=pltl.scatter(df_new[['Age']])
fig.write_html('Lof_scatter.html',auto_open=True)

fig=pltl.box(df_new[['Age']])
fig.write_html('Lof_box.html',auto_open=True)

df2_score=alg_lof.negative_outlier_factor_
# pltl.scatter(df2[["Age"]])
#pltl.scatter(title='Local Outlier Factor',color='k',size=3)
plt.title('Local Outlier Factor')
plt.scatter(df2.index, df2['Age'],color='k',s=3,label='Data points')
#radius=(df2_score.max()- df2_score)/(df2_score.max()-df2_score.min())
#radius=df2_score/df2_score.min()
radius=model
plt.scatter(df2.index,df2['Age'],s=50*radius, edgecolors='r',facecolors="None",
            label='Outliers')
plt.show()

#
# # LOF local outlier Factor
# # using Local Outlier Factor (LOF) using negative outlier factor score as radius
# #
# # The anomaly score of each sample is called Local Outlier Factor. It measures
# # the local deviation of density of a given sample with respect to its neighbors.
# # The anomaly score depends on how isolated the object is with respect to the
# # surrounding neighborhood. More precisely, locality is given by k-nearest
# # neighbors, whose distance is used to estimate the local density. By comparing
# # the local density of a sample to the local densities of its neighbors, one
# # can identify samples that have a substantially lower density than their
# # neighbors. These are considered outliers.
#
from sklearn.neighbors import LocalOutlierFactor
alg_lof=LocalOutlierFactor(n_neighbors=10,p=2)

# # use fit_predict to compute the predicted labels of the training samples
# # (when LOF is used for outlier detection, the estimator has no predict,
# # decision_function and score_samples methods).
model=alg_lof.fit_predict(df2[['Age']])
print(model)
non_outlier_list=model != -1
outlier_list=model ==-1
print('Length of outlier',sum(outlier_list))
print(non_outlier_list)
print('outlier_list',outlier_list)
print(df2.loc[non_outlier_list,'Age'])
print(df2.loc[outlier_list,'Age'])
print(df2.loc[outlier_list,'Age'].unique())
df_new=df2.loc[non_outlier_list,df2.columns]
print(df_new.head())
print(alg_lof.negative_outlier_factor_)
df2['Age_LoF']=alg_lof.negative_outlier_factor_.tolist()

fig=pltl.scatter(df2[['Age']],color=model)
fig.write_html('lof_scatter.html',auto_open=True)
# fig=pltl.scatter(df_new[['Age']],color=model)
# fig.write_html('lofscatter.html',auto_open=True)
sns.distplot(df_new['Age'])
plt.show()

fig=pltl.scatter(df_new[['Age']])
fig.write_html('Lof_scatter.html',auto_open=True)

fig=pltl.box(df_new[['Age']])
fig.write_html('Lof_box.html',auto_open=True)

df2_score=alg_lof.negative_outlier_factor_
# pltl.scatter(df2[["Age"]])
#pltl.scatter(title='Local Outlier Factor',color='k',size=3)
plt.title('Local Outlier Factor')
plt.scatter(df2.index, df2['Age'],color='k',s=3,label='Data points')
#radius=(df2_score.max()- df2_score)/(df2_score.max()-df2_score.min())
#using the negative_outlier_factor score as radius. This will gives the
#circle size which is bigger for out lier
radius=df2_score/df2_score.max()
#radius=model
plt.scatter(df2.index,df2['Age'],s=100*radius, edgecolors='r',facecolors="None",
            label='Outliers')
plt.show()
#
# Isolation forest’s basic principle is that outliers are few and far from
# the rest of the observations.using isolation forest One efficient way of
# performing outlier detection in high-dimensional datasets is to use random
# Isolation forests. The ensemble.IsolationForest ‘isolates’ observations by
# randomly selecting a feature and then randomly selecting a split value between
# the maximum and minimum values of the selected feature.
#
# Since recursive partitioning can be represented by a tree structure, the number
# of splittings required to isolate a sample is equivalent to the path length from
# the root node to the terminating node.
#
# This path length, averaged over a forest of such random trees, is a measure of
# normality and our decision function.
#
# Random partitioning produces noticeably shorter paths for anomalies. Hence,
# when a forest of random trees collectively produce shorter path lengths for
# particular samples, they are highly likely to be anomalies.

from sklearn.ensemble import IsolationForest
alg_Isoforest=IsolationForest(max_samples=4,n_estimators=90,random_state=42)
model=alg_Isoforest.fit(df2[['Age']])
print(model)
y_pred=model.predict(df2[["Age"]])
print(y_pred)
outlier_list=df2['Age'][y_pred==-1]
non_outlier_list=df2['Age'][y_pred==1]
print(df2['Age'][y_pred==1].index)
print(df2['Age'][y_pred==1])

print(outlier_list)
fig=pltl.scatter(df2[['Age']],color=y_pred)
fig.write_html('ISOforest_scatter.html',auto_open=True)
df_new=df2.loc[df2['Age'][y_pred==1].index,df2.columns]

sns.distplot(df_new['Age'])
plt.show()
radius=y_pred
plt.scatter(df2.index, df2['Age'],color='k',s=3,label='Data points')
plt.scatter(df2.index,df2['Age'],s=100*radius, edgecolors='r',facecolors="None",
             label='Outliers')
plt.show()