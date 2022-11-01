import optuna
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#
# df=pd.read_csv('Admission_Prediction.csv')
#
# print(df)
#
# #The data is of regression problem
# #Make sure that the data set is of without any null values
# print(df.describe())
# print(df.isnull().sum())
# df['GRE Score']=df['GRE Score'].fillna(df['GRE Score'].mean())
# df['TOEFL Score']=df['TOEFL Score'].fillna(df['TOEFL Score'].mean())
# df['University Rating']=df['University Rating'].fillna(df['University Rating'].mean())
# print(df.isnull().sum())
#
#
# print(df.isin([0,0.0]).sum())
# #Research column has zeroes so let's drop as of now
# #df=df.drop(columns=['Research'])
# x=df.drop(columns=['Serial No.','Chance of Admit'])
# y=df['Chance of Admit']
#
# #Apply standardization
# from sklearn.preprocessing import StandardScaler
# alg_scaler=StandardScaler()
# x=alg_scaler.fit_transform(x)
# # NVIDIA-SMI execute this command in the terminal to list out the GPU details of your system on the console
# #CUDA is driver program which helps you to execute your program in CPU to inside the GPU program.
# #Let us build a function for defining the parameters
#
# def objective(trail,data=x,target=y):
#     x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=.20,random_state=30)
#     param={
#         'tree_method':'gpu_hist' , # to execute your algorithm in GPU you need to use this method
#         'lambda':trail.suggest_loguniform('lambda',1e-4,10.0), #lambda responsible for regularisation, it will keep l2
#                         #regularization. In every trial suggest us loguniform (i.e.labmda value) any parameter
#                       #value which is less than 10.0)
#         'alpha':trail.suggest_loguniform('alpha',1e-4,10.0),
#         'colsample_bytree':trail.suggest_categorical('colsample_bytree',[.3,.4,.5,.6,.7,.8,.9,1]),
#         'subsample':trail.suggest_categorical('subsample',[.3,.4,.5,.6,.7,.8,.9,1]),
#         'learning_rate':trail.suggest_categorical('learning_rate',[.0001,.00004,.0005,.006,.007,.08,.9,1,10]),
#         'n_estimator':30000,
#         'max_depth':trail.suggest_categorical('max_depth',[3,4,5,6,7,8,9,10,11,12]),
#         'random_state':trail.suggest_categorical('ranodm_state',[10,20,30,200,300]),
#         'min_child_weight':trail.suggest_int('min_child_weight',1,200)
#
#         # to know the list of parameters we can provide xgb.XGBRegressor() do shift tab in jupyternotes to get the details
#     }
#
#     alg_xgb=xgb.XGBRegressor(**param)
#     alg_xgb.fit(x_train,y_train,eval_set=[(x_test,y_test)],verbose=True)
#     y_pred=alg_xgb.predict(x_test)
#     rmse=mean_squared_error(y_test,y_pred)
#     return rmse
#
#
# #Now we need to pass this objective to optuna which is a hyperparameter tuner
#
# find_param=optuna.create_study(direction='minimize')
# find_param.optimize(objective,n_trials=10)
# #print(find_param.best_trial.params)
# best_params=find_param.best_trial.params
# print('best_params:', best_params)
# #The above code will give you error as found NAN but there are no NAN in our data set, this error is due to some
# #cache in your system. After executing the same code mulitiple times this error will go automatically some times
# #and may re appear again for certain execution. This error is nothing to do with your data.
# #Once the above code executes successfully you will be able to see below output
# #output
# #parameters: {'lambda': 0.07440305134683776,
# # 'alpha': 5.20428507297435,
# # 'colsample_bytree': 0.3,
# # 'subsample': 0.5,
# # 'learning_rate': 0.006,
# # 'max_depth': 12,
# # 'ranodm_state': 300,
# # 'min_child_weight': 84}. Best is trial 6 with value: 0.008519613369798333.
# print(find_param.trials_dataframe()) #this will give the details about what are all the trials it has done
# # how much time each trial has taken, what are all the parameters and parameters combinations it has tested
# #those details it will give.
# #To see what are all the optimization it has taken those details can be obtained by below code
# #print(optuna.visualization.plot_optimization_history)
# fig=optuna.visualization.plot_optimization_history(find_param)#This gives the details of for each of the trails
# #what are the objective values (such as error, accuracy values) and what is the number of trials.
# fig.show()
# fig=optuna.visualization.plot_slice(find_param)#To find out the visualization of each and every parameter and want
# #to see how each parameter is behaving with resepect to trials.
# fig.show()
# fig=optuna.visualization.plot_contour(find_param,params=['alpha','lambda']) #this shows how parameters are moving
#           #with respect to each other parameters.
# fig.show()
# #optuna.visualization.
#
# #optuna uses plotly internally
# #plt.show()
#
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=30)
# alg_xgb_final=xgb.XGBRegressor(**best_params)
# alg_xgb_final.fit(x_train,y_train)
# print(alg_xgb_final.score(x_train,y_train))
# print(alg_xgb_final.score(x_test,y_test))
#
# #what is difference between optuna and gridsearchCV.
# #Here if you look in to the data frame which is given by optuna you are able to see that it is not considering
# #all the trials, what it is trying to do is , it is trying to select some of the parameters out of the parameters
# #given by us and only for some of the parameters it is trying to execute. Where as in grid search cv it will try
# #to take permutation combination of each and every possible parameters values, and it will try to give the result
# #based on all the possible trials.Optuna is just a library, your base module is not going to change. At the end
# #of the day who is giving you the best model is your base model.For example if we are using the XGbooster as a
# #base model then XGBooster is going to provide the best parameters.
# #Wheather it is grid search cv, random search cv, optuna, teapot all these are helping you to select the best
# #parameters. There are other hyper parameter tuning mechanisms, such as sequential model optimizer or baysian
# #optimizer. All theser are doing the same thing, may be each have there own wrappers (some one have coded to
# #optimize the base method)
# #Why optuna is popular? Optuna supports Keras, tenserflow,pytorch, chainer, XGBOOST, Sklearn libraries.
# #It supports almost each and every library, even NLP libraries, such as catalist, fast chainer, AI etc
# #Keras itself has a kerastuner, but it is hard to implement, so we can use optuna easily.
# #XGBOOST is using the tree based regression as base model, i.e why we are able to control the tree depth,
# #leaf node, weightage...etc.

#XGBOOSTER classification problem with using optuna.
df=pd.read_csv('winequality_red.csv')
print(df)
print(df.isnull().sum())
x=df.drop(columns=['quality'])
y=df.quality
print(y)
#XGBOOST classifier is unable to infer the classification values if it is not starting from 0,1,2,3...etc
#we need to do label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = pd.DataFrame(le.fit_transform(y))
print(y)

def objective_classification(trial,data=x,target=y):
    x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=.20,random_state=30)
    param={
        'tree_method':'gpu_hist',
        'verbosity':3,
        'objective':'binary:logistics',
        'booster':trial.suggest_categorical('booster',['dart','gbtree','gblinear']),
        'lambda':trial.suggest_float('lambda',1e-4,1),
        'alpha':trial.suggest_float('alpha',1e-4,1),
        'subsample':trial.suggest_float('subsample',.1,.5),
        'colsample_bytree':trial.suggest_float('colsample_bytree',.1,.5)

    }
    if param['booster'] in ['gbtree','dart']:
        param['gamma']:trial.suggest_float('gamma',1e-3,4)
        param['eta']:trial.suggest_float('eta',.001,5)
    # similarly add parameters for different different boosters.

    alg_xgb_class=xgb.XGBClassifier(**param)
    alg_xgb_class.fit(x_train,y_train,eval_set=[(x_test,y_test)])


    y_pred=alg_xgb_class.predict(x_test)
    print(alg_xgb_class.score(x_train,y_train))
    print(alg_xgb_class.score(x_test,y_test))
    accuracy=alg_xgb_class.score(x_test,y_test)
    return accuracy

xgb_classi_optuna=optuna.create_study(direction='minimize')
xgb_classi_optuna.optimize(objective_classification,n_trials=10)

print(xgb_classi_optuna.best_trial)
print(xgb_classi_optuna.best_trial.params)
best_params=xgb_classi_optuna.best_trial.params
print('best one',best_params)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=30)
alg_xgb_class_final=xgb.XGBClassifier(**best_params)
alg_xgb_class_final.fit(x_train,y_train)
print('Score',alg_xgb_class_final.score(x_test,y_test))

