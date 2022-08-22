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

df1=df[['Fare',"Age"]]
print(min(df1['Fare']))
print(max(df1['Fare']))
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
df2=pd.DataFrame(alg_minmax.transform(df1),columns=['Fare','Age'])
print(df2)

#Now we can see the change in the data range min max range..
#
print(df1.describe())
print(df2.describe())


#Min-Max scaling (change range 0-5)

alg_minmax=MinMaxScaler(feature_range=(0,5))
alg_minmax.fit(df1)
print(df1)
print(alg_minmax.transform(df1))
df2=pd.DataFrame(alg_minmax.transform(df1),columns=['Fare','Age'])
print(df2)
print(df1.describe())
print(df2.describe())

import seaborn as sns
fig=pltl.histogram(df1,title='distribution before min max scaling')
fig.write_html('hist.html',auto_open=True)

fig=pltl.histogram(df2,title='distribution after min max scaling')
fig.write_html('hist_min_max.html',auto_open=True)

fig=pltl.histogram(df1[['Fare']],title='Fare distribution before scaling')
fig.write_html('Fare_hist.html',auto_open=True)

fig=pltl.histogram(df2[['Fare']],title='Fare distribution after min max scaling')
fig.write_html('Fare_hist_min_max.html',auto_open=True)

fig=pltl.histogram(df1[['Age']],title='Age distribution before scaling')
fig.write_html('Age_hist.html',auto_open=True)

fig=pltl.histogram(df2[['Age']],title='Age distribution after min max scaling')
fig.write_html('Age_hist_min_max.html',auto_open=True)

fig=pltl.box(df1,title='Fare box plot before scaling')
fig.write_html('box.html',auto_open=True)

fig=pltl.box(df2,title='Age box after min max scaling')
fig.write_html('box_min_max.html',auto_open=True)

#Standard Scalar method
alg_scalar=StandardScaler()
print(alg_scalar.fit_transform(df1))
df2=pd.DataFrame(alg_scalar.fit_transform(df1),columns=['Fare','Age'])

print(df1)
print(df2)
print(df1.describe())
print(df2.describe())

import seaborn as sns
fig=pltl.histogram(df1,title='distribution before standard scalar')
fig.write_html('Dist.html',auto_open=True)
fig=pltl.histogram(df2,title='distribution after standard scalar')
fig.write_html('Dist_Standard_scalar.html',auto_open=True)

fig=pltl.histogram(df1[['Fare']],title='Fare distribution before standard scalar')
fig.write_html('Fare_Dist.html',auto_open=True)

fig=pltl.histogram(df2[['Fare']],title='Fare distribution after standard scalar')
fig.write_html('Fare_Dist_Standard_scale.html',auto_open=True)

fig=pltl.histogram(df1[['Age']],title='Age distribution before standard scalar')
fig.write_html('Age_dist.html',auto_open=True)

fig=pltl.histogram(df2[['Age']],title='Age distribution after standar scalar')
fig.write_html('Age_dist_standard_scale.html',auto_open=True)

fig=pltl.box(df1,title='Box plots before standard scalar')
fig.write_html('Box_plots.html',auto_open=True)

fig=pltl.box(df2,title='Box plots after standard scalar')
fig.write_html('Box_standard_scale.html',auto_open=True)

#Robust scalar

alg_robust=RobustScaler()
print(alg_robust.fit_transform(df1))
df2=pd.DataFrame(alg_robust.fit_transform(df1),columns=['Fare','Age'])

print(df1)
print(df2)
print(df1.describe())
print(df2.describe())
fig=pltl.histogram(df1,title='distribution before robust scaling')
fig.write_html('hist3.html',auto_open=True)

fig=pltl.histogram(df2,title='distribution after robust scaling')
fig.write_html('hist_robust.html',auto_open=True)

fig=pltl.histogram(df1[['Fare']],title='Fare distribution before robust scaling')
fig.write_html('Fare_hist3.html',auto_open=True)

fig=pltl.histogram(df2[['Fare']],title='Fare distribution after robust scaling')
fig.write_html('Fare_hist_robust.html',auto_open=True)

fig=pltl.histogram(df1[['Age']],title='Age distribution before robust scaling')
fig.write_html('Age_hist3.html',auto_open=True)

fig=pltl.histogram(df2[['Age']],title='Age distribution after robust scaling')
fig.write_html('Age_hist_robust.html',auto_open=True)

fig=pltl.box(df1,title='Fare box plot before robust scaling')
fig.write_html('box3.html',auto_open=True)

fig=pltl.box(df2,title='Age box after robust scaling')
fig.write_html('box_robust.html',auto_open=True)


#Power Transform-Yeo-Johnson
from sklearn.preprocessing import PowerTransformer
from matplotlib import pyplot
alg_transform=PowerTransformer(method='yeo-johnson',standardize=True)


df2=pd.DataFrame(alg_transform.fit_transform(df1),columns=['Fare','Age'])

pyplot.hist(df1,bins=25)
pyplot.title(label='Distribution before power Transform')
pyplot.show()
pyplot.hist(df2,bins=25)
pyplot.title(label='Distribution after power Transform')
pyplot.show()

fig=pltl.box(df1,title='Age box before powertransform')
fig.write_html('box_4.html',auto_open=True)
fig=pltl.box(df2,title='Age box after powertransform')
fig.write_html('box_powertransform.html',auto_open=True)

# #Power Transform-Box Cox
from sklearn.preprocessing import PowerTransformer
from matplotlib import pyplot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
alg_Box_cox=PowerTransformer(method='box-cox',standardize=True)
scaler=MinMaxScaler(feature_range=(1,2))
pipeline=Pipeline(steps=[('s',scaler),('p',alg_Box_cox)])
df2=pd.DataFrame(pipeline.fit_transform(df1),columns=['Fare','Age'])

pyplot.hist(df1,bins=25)
pyplot.title(label='Distribution before Box_COX transform')
pyplot.show()
pyplot.hist(df2,bins=25)
pyplot.title(label='Distribution after BOX-COX transform')
pyplot.show()

fig=pltl.box(df1,title='Box plot Before BOX-COX power transform')
fig.write_html('Box_5.html',auto_open=True)
fig=pltl.box(df2,title='Box plot After BOX-COX power Transform')
fig.write_html('BOxplot_BOX_COX_Powertransform.html',auto_open=True)

fig = df1.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
pyplot.show()


fig = df2.hist(xlabelsize=4, ylabelsize=4)
[x.title.set_size(4) for x in fig.ravel()]
pyplot.show()

#log transform
import numpy as np
from matplotlib import pyplot
df2=df1.copy()
df2['Age']=np.log(df1['Age'])
df2['Fare']=np.log(df1['Fare'])
print(df1)
print(df2)

fig=pltl.box(df1,title='Box plot before log transform')
fig.write_html('Box_6.html',auto_open=True)

fig=pltl.box(df2,title='Box plot After Log transform')
fig.write_html('Box_Log_transform.html',auto_open=True)

fig=pltl.histogram(df1,title="Distribution before log transform")
fig.write_html('Hist_before.html',auto_open=True)

fig=pltl.histogram(df2,title='Distribution after Log transform')
fig.write_html('Hist_Log_transform.html',auto_open=True)

#Quantile Transforms

from sklearn.preprocessing import QuantileTransformer

alg_Quantile=QuantileTransformer(output_distribution='normal')
df2=pd.DataFrame(alg_Quantile.fit_transform(df1))

fig=pltl.histogram(df1,title='Histogram before Quantile transform')
fig.write_html('Histo_before_Quantile.html',auto_open=True)

fig=pltl.histogram(df2,title='Histogram after quantile transform')
fig.write_html('Histogram_quantile_transform.html',auto_open=True)

fig=pltl.box(df1,title='Box before Quantile transform')
fig.write_html('Box_7.html',auto_open=True)

fig=pltl.box(df2,title='Boxplot after Quantile Transform')
fig.write_html('Box_Quantile.html',auto_open=True)


#Guassian Transformation

import scipy.stats as stat
import pylab
plt.figure(figsize=(10,6))
# QQ-PLOT to check wheather the distribution is normal or not=stat.probplot with dist as norm and
#plot as pylab
stat.probplot(df1['Fare'],dist='norm',plot=pylab)
plt.show()
stat.probplot(df1['Age'],dist='norm',plot=pylab)
plt.show()


# #Reciprocal transformation
import scipy.stats as stat
import pylab
df2=df1.copy()
df2['Age']=1/df1['Age']
df2['Fare']=1/df1['Fare']
stat.probplot(df2['Fare'],dist='norm', plot=pylab)
plt.show()

stat.probplot(df2['Age'],dist='norm',plot=pylab)
plt.show()

fig=pltl.box(df1,title='Box plot before reciprocal transform')
fig.write_html('Box_8.html',auto_open=True)

fig=pltl.box(df2,title="Box plot After reciprocal transform")
fig.write_html('Box_reciprocal_transform.html',auto_open=True)

#Square root transfermation
import scipy.stats as stat
import pylab

df2=df1.copy()
df2['Fare']=df1['Fare']**(1/2)
df2['Age']=df1['Age']**(1/2)

stat.probplot(df2['Fare'],dist='norm',plot=pylab)
plt.show()

stat.probplot(df2['Age'],dist='norm',plot=pylab)
plt.show()

fig=pltl.box(df1,title='Box plot before square root transformation')
fig.write_html('Box_9.html',auto_open=True)

fig=pltl.box(df2,title='Box plot after square root transformation')
fig.write_html('Box_Square_transformation.html',auto_open=True)


fig=pltl.histogram(df1,title='hist plot before square root transformation')
fig.write_html('hist_9.html',auto_open=True)

fig=pltl.histogram(df2,title='hist plot after square root transformation')
fig.write_html('hist_Square_transformation.html',auto_open=True)



#
# #Exponential transfermation
import scipy.stats as stat
import pylab

df2=df1.copy()
df2['Fare']=df1['Fare']**(1/1.2)
df2['Age']=df1['Age']**(1/1.2)

stat.probplot(df2['Fare'],dist='norm',plot=pylab)
plt.show()

stat.probplot(df2['Age'],dist='norm',plot=pylab)
plt.show()

fig=pltl.box(df1,title='Box plot before Exponential transformation')
fig.write_html('Box_10.html',auto_open=True)

fig=pltl.box(df2,title='Box plot after Exponential  transformation')
fig.write_html('Box_Exponential _transformation.html',auto_open=True)


fig=pltl.histogram(df1,title='hist plot before Exponential  root transformation')
fig.write_html('hist_10.html',auto_open=True)

fig=pltl.histogram(df2,title='hist plot after Exponential  transformation')
fig.write_html('hist_Exponential _transformation.html',auto_open=True)



#Normalizer



from sklearn.preprocessing import Normalizer
import scipy.stats as stat
import pylab

alg_Normalizer=Normalizer(norm='l2')

df2=pd.DataFrame(alg_Normalizer.fit_transform(df1),columns=['Fare','Age'])
print(df2)
stat.probplot(df2['Fare'],dist='norm',plot=pylab)
plt.show()

stat.probplot(df2['Age'],dist='norm',plot=pylab)
plt.show()

fig=pltl.box(df1,title='Box plot before Normalizer transformation')
fig.write_html('Box_11.html',auto_open=True)

fig=pltl.box(df2,title='Box plot after Normalizer  transformation')
fig.write_html('Box_Normalizer _transformation.html',auto_open=True)


fig=pltl.histogram(df1,title='hist plot before Normalizer  root transformation')
fig.write_html('hist_11.html',auto_open=True)

fig=pltl.histogram(df2,title='hist plot after Normalizer  transformation')
fig.write_html('hist_Normalizer _transformation.html',auto_open=True)


#MaxAbsScaler

from sklearn.preprocessing import MaxAbsScaler
import scipy.stats as stat
import pylab

alg_MaxAbsScaler=MaxAbsScaler()

df2=pd.DataFrame(alg_MaxAbsScaler.fit_transform(df1),columns=['Fare','Age'])
print(df2)
stat.probplot(df2['Fare'],dist='norm',plot=pylab)
plt.show()

stat.probplot(df2['Age'],dist='norm',plot=pylab)
plt.show()

fig=pltl.box(df1,title='Box plot before MaxAbsScaler transformation')
fig.write_html('Box_12.html',auto_open=True)

fig=pltl.box(df2,title='Box plot after MaxAbsScaler transformation')
fig.write_html('Box_MaxAbsScaler _transformation.html',auto_open=True)


fig=pltl.histogram(df1,title='hist plot before MaxAbsScaler  root transformation')
fig.write_html('hist_12.html',auto_open=True)

fig=pltl.histogram(df2,title='hist plot after MaxAbsScaler transformation')
fig.write_html('hist_MaxAbsScaler _transformation.html',auto_open=True)