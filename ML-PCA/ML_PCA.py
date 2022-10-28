import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from numpy.linalg import eig
from sklearn.decomposition import PCA


#Two dimentiondata take
data=np.array([[3,4],[2,8],[6,9],[10,12]])
print(data)

#Create data frame
df=pd.DataFrame(data,columns=['ML','DL'])

print(df)

plt.scatter(df['ML'],df['DL'])
plt.show()

#PCA Steps
#standardization
#Covariance
#Eign value and EIGN vector
#PC

#Transpose the array (i.e. rows to column and columns to row)
#data.T

#Get the mean of each column in array

data_mean_by_column=np.mean(data,axis=0)
print(data_mean_by_column)

#Get the zerocentric data manually

data_zero_centric=data-data_mean_by_column

print(data_zero_centric)

#Step-2 find the co variance of the zero centric data with Transpose of the data
data_cov=np.cov(data_zero_centric.T)
print(data_cov)

#Find the Eign value adn Eign vector
eign=np.linalg.eig(data_cov)
print(eign)
eign_values=eign[0]
eign_vectors=eign[1]
print(eign_values)
print(eign_vectors)

#Now find out the principle component by dot product of zero centric data with the eign vecotrs
pc=eign_vectors.T.dot(data_zero_centric.T).T

print(pc) #We are getting 2 principal components
alg_pca=PCA()
pcs=alg_pca.fit_transform(data_zero_centric) #Through this way also we age getting the same PCs
print(pcs)
df_pcs=pd.DataFrame(pcs,columns=['PC1','PC2'])

print(df_pcs)

#If you do the inverse transform of pcs we will get the zero_centric data back
inverse_data=alg_pca.inverse_transform(pcs)
print(inverse_data)    #Inverse data is same as our zerocentric data
print(data_zero_centric)

#Now let us find out the variation of data covered by these two PCs
print(alg_pca.explained_variance_ratio_)
#From the above variance ratio we can observe that the PC1 coverse 90.42% of daat and PC2 coverse only 9.5% of
#variation of the data. So based on this we will choose the principal components which represents maximum % of
#data. i.e. in this case PC1

gl_data_1=pd.read_csv('glass.data')
print(gl_data_1)
print(gl_data_1.shape)
gl_data=gl_data_1.drop(['index','Class'],axis=1)
print(gl_data.isnull().sum())
print(gl_data.describe().T)

#PCA Steps
#Standardization
from sklearn.preprocessing import StandardScaler
alg_scale=StandardScaler()
scaled_data=alg_scale.fit_transform(gl_data)
print(scaled_data)
scaled_df=pd.DataFrame(scaled_data)
print(scaled_data)
print(scaled_df.describe())

from sklearn.decomposition import PCA
alg_pca2=PCA()
pcs=alg_pca2.fit_transform(scaled_df)

print(pcs)
pcs_df=pd.DataFrame(pcs)
print(pcs_df)
print(scaled_df.shape)
print(pcs_df.shape)
variations=alg_pca2.explained_variance_ratio_
print(variations)
print(max(variations))
print(min(variations))
print(sum(variations))

plt.figure()
plt.plot(np.cumsum(variations))
plt.xlabel('number of components')
plt.ylabel('Variance')
plt.title('pca_representation')
plt.show()
print(sorted(variations, reverse=True))
print(sum(sorted(variations,reverse=True)[:7]))


reduced_pca=PCA(n_components=7)
red_pcs=reduced_pca.fit_transform(scaled_df)
print(scaled_df)
reduction_df=pd.DataFrame(red_pcs,columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7'])
print(reduction_df)

#Let us perfome on classification algorithm
X=reduction_df
y=gl_data_1['Class']
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
alg_dicision=DecisionTreeClassifier()
alg_dicision.fit(X,y)
# User will be providing the origional form of data, so we need to perform standard scaleing and PCA steps for
# the user data as well.
#print(gl_data.iloc[0]). This way you need to build the prediction pipeline as well.
test_data=pd.DataFrame(gl_data.iloc[209]).T #data is in series index and single column so need to perform Transpose
print('test data \n', test_data)
scaled_test_data_df=pd.DataFrame(alg_scale.transform(test_data))
print(scaled_test_data_df)
test_data_df=pd.DataFrame(reduced_pca.transform(scaled_test_data_df))
print(alg_dicision.predict(test_data_df))









