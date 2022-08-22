import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as pltl
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import
df=pd.read_csv('titanic.csv')
print(df.head())
print(df.columns)

df1=df[["Age"]]
# print(min(df1['Fare']))
# print(max(df1['Fare']))
print(min(df1['Age']))
print(max(df1['Age']))

print(df1["Age"].isnull().sum())
df1=df1.dropna()
print(df1.shape)

#Min-Max scaling (Default range 0-1)

alg_minmax=MinMaxScaler()
alg_minmax.fit(df1)
print(df1)
print(alg_minmax.transform(df1))
df2=pd.DataFrame(alg_minmax.transform(df1),columns=['Age'])
print(df2)

#Now we can see the change in the data range min max range..

print(df1.describe())
print(df2.describe())


#Min-Max scaling (change range 0-5)

# alg_minmax=MinMaxScaler(feature_range=(0,5))
# alg_minmax.fit(df1)
# print(df1)
# print(alg_minmax.transform(df1))
# df2=pd.DataFrame(alg_minmax.transform(df1),columns=['Fare','Age'])
# print(df2)
# print(df1.describe())
# print(df2.describe())

import seaborn as sns
# sns.distplot(df1['Fare'])
# plt.suptitle('Fare is rightly skewed')
#
# plt.savefig('Fare_dist.png')
#
# sns.distplot(df2['Fare'])
# plt.suptitle('Fare distribution after min max scale transformation')
# plt.savefig('Fare_dist_MinMax scale.png')
#plt.show()


# sns.distplot(df1['Age'])
# plt.suptitle('Age Distribution before min max scaling')
# plt.savefig('Age_dist.png')
print(df2.head())
sns.distplot(df2['Age'])
plt.suptitle('Age Distribution after min max scaling')
plt.savefig('Age_Dist_minmax.png')

# sns.boxplot(df1['Age'])
# plt.suptitle('Age box before scaling')
# plt.savefig('Age_box.png')

sns.boxplot(df2['Age'])
plt.suptitle('Age box after minmax scaling')
plt.savefig('Age_box_minmax.png')

fig=pltl.box(df2[['Age']])
fig.write_html('Age_box.html',auto_open=True)
#
# df1.plot(kind='box',title='boxplot')
# plt.show()
# df2.plot(kind='box',title='boxplot')
# plt.show()
# #
#
# #Standard Scalar method
# alg_scalar=StandardScaler()
# print(alg_scalar.fit_transform(df1))
# df2=pd.DataFrame(alg_scalar.fit_transform(df1),columns=['Fare','Age'])
#
# print(df1)
# print(df2)
# print(df1.describe())
# print(df2.describe())
#
# import seaborn as sns
# sns.distplot(df1['Fare'])
#
# plt.show()
# sns.distplot(df1['Age'])
# plt.show()
#
# sns.distplot(df2['Fare'])
# plt.show()
#
# sns.distplot(df2['Age'])
# plt.show()
#
# df1.plot(kind='box',title='Box plot' )
# plt.show()
#
# df2.plot(kind='box',title='Box plot')
# plt.show()
#
# alg_robust=RobustScaler()
# print(alg_robust.fit_transform(df1))
# df2=pd.DataFrame(alg_robust.fit_transform(df1),columns=['Fare','Age'])
#
# print(df1)
# print(df2)
# print(df1.describe())
# print(df2.describe())
#
# import seaborn as sns
# sns.distplot(df1['Fare'])
# plt.show()
#
# sns.distplot(df2['Fare'])
# plt.show()
#
# sns.distplot(df1['Age'])
# plt.show()
#
# sns.distplot(df2['Age'])
# plt.show()
#
# print(df1)
# print(df2)
# print(df1.describe())
# print(df2.describe())
#
# df1.plot(kind='box',title='Boxplot')
# plt.show()
#
# df2.plot(kind='box',title='Boxplot')
# plt.show()