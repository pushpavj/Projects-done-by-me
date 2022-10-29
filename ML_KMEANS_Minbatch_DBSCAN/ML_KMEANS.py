import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv('https://gist.githubusercontent.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf/raw/8bd6144a87988213693754baaa13fb204933282d/Mall_Customers.csv')
print(df)

#Let us perform grouping of the customers based on Annual Income and Spending score

x=df.iloc[:,3:]
x2=df.iloc[:,3:]
x3=df.iloc[:,3:]
print(x)

#Let us find the number of clusters k is required
wcss=[]

for i in range(1,15):
    alg_kmean=KMeans(n_clusters=i,init='k-means++',random_state=30)
    alg_kmean.fit(x)
    wcss.append(alg_kmean.inertia_) # inertia is = within cluster summation square)

print(wcss)
#Let us build the elbow graph to find the number of k
plt.plot(range(1,15),wcss)
plt.show()

#From the elbow graph we can find the k as 5

alg_Kmean2=KMeans(n_clusters=5,init='k-means++',random_state=30)
alg_Kmean2.fit_predict(x)

x['cluster_numbers']=alg_Kmean2.fit_predict(x)
print(x)

print(x[x['cluster_numbers']==4])

print(x[x['cluster_numbers']==0])

print(alg_Kmean2.predict([[55,31]]))

#Let us see using mini batch Kmean

from sklearn.cluster import MiniBatchKMeans
alg_minibatch=MiniBatchKMeans(n_clusters=5)
alg_minibatch.fit_predict(x2)

x2['mini_labels']=alg_minibatch.fit_predict(x2)
print(x2)
print(x2[x2['mini_labels']==2])
print(alg_minibatch.predict([[55,31]]))

#Let us perform using DBSCAN

from sklearn.cluster import DBSCAN
alg_dbscan=DBSCAN(eps=1,min_samples=3)
alg_dbscan.fit(x3)
print(alg_dbscan.labels_)
#Total number of clusters
print(set(alg_dbscan.labels_))
#Total lenght of the clusters
print(len(set(alg_dbscan.labels_)))
#Group -1 indicates that it is a noise or outlier where it is not able to accupi it to any of the cluster

x3['dbscan_labels']=alg_dbscan.labels_
print(x3)

#Let us see the accuracy of the cluster algorithm

from sklearn.metrics import jaccard_score,accuracy_score,rand_score,adjusted_rand_score

#Take Kmean out put as client label or ground truth table
print(jaccard_score(x['cluster_numbers'],x3['dbscan_labels'],average='macro'))












