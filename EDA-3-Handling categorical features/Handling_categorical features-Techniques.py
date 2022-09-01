# #Handling categorical features
#
import pandas as pd
#
#one hot encoding method using pd.get_dummies(df) will create the on hot encoding for categorical variables.
#dis advantages of one hot encoding are it creates more number of features as it create new-feature
#for each category which leads to curse of dimentionality. i.e, more number of features leads to overfitting
#
df=pd.read_csv('titanic.csv',usecols=['Sex'])
print(pd.get_dummies(df))
#it will create the two features one for male and another for female
#Since there are only two categories in the Sex feature, having only one category feature of one hot encoding is
#fine to represent both of them, so for that we can drop one of its feature as below
print(pd.get_dummies(df,drop_first=True))
df=pd.read_csv('titanic.csv',usecols=['Embarked'])
print(df['Embarked'].unique())
df.dropna(inplace=True)
print(pd.get_dummies(df))
#It will create three one hot coded features
#Here also with haiving only two one hot coded feature we can represent the third one hot coded feature
#i.e. having 0 for two one hot coded features will represent that the third one will be having 1 in it
# so we can drop or remove one of the one hot coded features among the three.
print(pd.get_dummies(df,drop_first=True))

#Handling the feature with lot many categories with in it by using one hot encoding method

df=pd.read_csv('mercedes.csv')
print(df.head())
df=pd.read_csv('mercedes.csv',usecols=['X0','X1','X2',"X3",'X4','X5','X6'])
print(df.head())

print(df['X0'].value_counts())
# Let us see the total number of categories in each of the columns
for i in df.columns:
    print(df[i].unique())
    print(len(df[i].unique()))

#since the number of categories in each feature are very large, if we perform one hot encoding
#it will be creating that many new features which will be a problem.
#so in order to handle such features, now we will apply new type of one hot encoding. In this we will
#consider only the top 10 categories for converting them to one hot encoding, remaining features will be
#discarded or dropped or skipped.

print(df.X0.value_counts())
print(df.X0.value_counts().sort_values(ascending=False).head(10))
print(df.X0.value_counts().sort_values(ascending=False).head(10).index)

lst_10=df.X0.value_counts().sort_values(ascending=False).head(10).index
print(lst_10)
print(list(lst_10))
lst_10=list(lst_10)

#Let us do the on hot encoding for only the selected categories using numpy where clause
import numpy as np
for i in lst_10:
    df['X0_'+i]=np.where(df['X0']==i,1,0) #This will create the new variable for each short listed
    # categories in the lst_10 and each will have value as 1 only when df[X0] has value as that category
    #other wise it will have value as zero for the rest

print(df.columns)
print(df.head(100))


#Ordinal Number encoding-
import datetime

today_date=datetime.datetime.today()
print(today_date)

previous_date=today_date-datetime.timedelta(1)  #Delta gives the number of days in a month for calculations
print(previous_date)

#list comprahension to create 15 days date data
print([today_date - datetime.timedelta(x) for x in range(0,15)])
#let us convert these dates into data frame
days=[today_date - datetime.timedelta(x) for x in range(0,15)]
import pandas as pd
data=pd.DataFrame(days)
data.columns=['Days']

print(data.head())
print(data['Days'])

print(data['Days'].dt.day_name())
print(data['Days'].dt.day)
print(data['Days'].dt.month)
print(data['Days'].dt.year)
print(data['Days'].dt.month_name())
print(data['Days'].dt.minute)
print(data['Days'].dt.week)
print(data['Days'].dt.hour)

dictionary={'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
data['Weekday']=data['Days'].dt.day_name()
print(data.head())
data['Day_ordinal']=data['Weekday'].map(dictionary)
print(data.head())
# This is called as ordinal number encoding

#Count or Frequency encoding
import pandas as pd
train_set=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',header=None,
                      index_col=None)
print(train_set.head())
cat_columns=[1,3,5,6,7,8,9,13]
print(train_set[cat_columns])

#Let us have only categorical columns in our data set
train_set=train_set[cat_columns]
#Let us have some feature name for each of the columns
train_set.columns=['Employment','Degree','Status','Designation','Familiy_job','Race','Sex','country']
print(train_set.head())

#Let us find the number of unique categories in each of the feature
for feature in train_set.columns:
    print(feature,':',len(train_set[feature].unique()),'labels')
#We can find that country has highest number of categories, let us apply count of frequency method on
#country column

print(train_set['country'].value_counts())
#we can either drop the column having ? or we can replace it as other category etc..
#Let us convert this data in to dictionary form
country_map=train_set['country'].value_counts().to_dict()
#Now let us replace the cortry field value with number of occurence of that country instead of country name
# this is called as Frequency Count encoding

train_set['country']=train_set['country'].map(country_map)
#The map function will map the value in data set with key inside the dictionary and gives the corresponding
#value in dictionary as output

print(train_set['country'].head(100))
#Advantages of Frequency count encoding are
#Easy to use, we are not increasing any feature space
#Dis advantage it will provide the same wight if the frequencies are same or it will not be able to
#distinguish between two countries data


#Target guided ordinal encoding
#.i.e giving the weightage to the catogorical data based on the hieghest value of the Target
#in this we will be oredering the labels according to the target
#Or replace the labels by the joint probability of being 1 or 0
import pandas as pd

df=pd.read_csv('titanic.csv',usecols=['Cabin','Survived'])
print(df.head())

#Let us replace the NAN values before applying the encoding
df['Cabin'].fillna('Missing',inplace=True)
print(df.head())

#Now let us consider the first character of the category
print(df['Cabin'].astype(str))
print(df['Cabin'].astype(str).str[0])
df['Cabin']=df['Cabin'].astype(str).str[0]
print(df['Cabin'].unique())
#Now let us see how many peoples are survived in each category of the Cabin
#print('test',df.groupby(df['Cabin'].mean()))
print(df.groupby(df['Cabin'])['Survived'].mean())
#Here it groups the out put by cabin and calculates the mean of the survived(which is a target variable)
# for each cabin
#.i.e giving the weightage to the catogorical data based on the hieghest value of the Target
print(df.groupby(df['Cabin'])['Survived'].mean().sort_values())
print(df.groupby(df['Cabin'])['Survived'].mean().sort_values().index)
#Let us store these labels as ordinal labels
ordinal_labels=df.groupby(df['Cabin'])['Survived'].mean().sort_values().index
print(ordinal_labels)
print(type(ordinal_labels))
#Let us give weightage to these labels accordingly
enumerate(ordinal_labels,0)
ordinal_label_2={j:i for i, j in enumerate(ordinal_labels,0)}
print(ordinal_label_2)
df['Cabin_ordinal_labels']=df['Cabin'].map(ordinal_label_2)
print(df)
#Now we can drop the Cabin column

#Mean encoding : it is similar to Target guided ordinal encoding, but the difference is here
#Instead of ordering the catogorical variable based on the mean of the target here we will replace the
#categorical variable with the mean of the Target it self.
import pandas as pd
df=pd.read_csv('titanic.csv',usecols=['Cabin','Survived'])
print(df.head())
#First let us replace the Nan value with missing category
df['Cabin']=df['Cabin'].fillna('missing')
print(df.head())
#Now let us consider the first character of each category
print(df['Cabin'].astype(str).str[0])
df['Cabin']=df['Cabin'].astype(str).str[0]
print(df['Cabin'].unique())
#Let us find the mean of the Target variable i.e. Survived for each of the Cabin type
print(df.groupby(df['Cabin'])['Survived'].mean())
ordinal_label=df.groupby(df['Cabin'])['Survived'].mean().sort_values()
print(ordinal_label)

ordinal_label_dict=ordinal_label.to_dict()
print(ordinal_label_dict)
df['Cabin_mean']=df['Cabin'].map(ordinal_label_dict)
print(df.head())
#Now we can drop the Cabin column
#Adantages is captures information within the label or Target hence it helps in
# rendering to new features as more predictable
# #it creates the monotonic relationship between variable and the target
#Dis advantage is It prones to overfitting

#Probability Ratio Encoding
#We will find out the probability of Target (survivied) based on the Cabin (categorical variabl)
#We will calculate the probobility of Not of Target (Died) as 1-probability of Target(Survivied)
#We will find the ratio as probability of Survived / probability of Died
#We will create the dictionary of the Probability ration with respect to Cabin
#We will create the new feature in main data set with mapping the Cabin with Probabilty Ratio
#We can drop the Cabin column
import pandas as pd

df=pd.read_csv('titanic.csv',usecols=['Cabin','Survived'])
print(df.head())
df['Cabin']=df['Cabin'].fillna('Missing')
print(df.head())
print(df['Cabin'].unique())

#Let us take the first character of each category
df['Cabin']=df['Cabin'].astype(str).str[0]
print(df.head())
print(df['Cabin'].unique())

print(df.groupby(df['Cabin'])['Survived'].mean())
#This gives the probability of the survieved for each cabin type
df_prob=pd.DataFrame((df.groupby(df['Cabin'])['Survived'].mean()))
print(df_prob)
#Now let us find the probability of the died for each cabin type
df_prob['Died']=1-df_prob['Survived']
print(df_prob)
df_prob['Probability_ratio']=df_prob['Survived']/df_prob['Died']
print(df_prob.head())
#Let us create dictionary
prob_dict=df_prob['Probability_ratio'].to_dict()
#Let us map to the titanic data set
df['Cabin_prob_ratio']=df['Cabin'].map(prob_dict)
print(df.head())
#Now we can drop the Cabin column