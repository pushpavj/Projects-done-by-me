#import sklearn as sns
import numpy as np
import scipy
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator
import matplotlib
import seaborn
#import EDA.py




'''This module is to build a ML model to predict whether income exceeds $50K/yr based on census data'''
def getprofilereport(df,reportname):
    profile = ProfileReport(df)
    profile.to_file(reportname)

def onehat(dframe):
    cate_cols = [i for i in dframe.columns if dframe[i].dtype == 'O']
    df1 = pd.get_dummies(dframe, columns=cate_cols)
    return cate_cols, df1
def cattonum(df):
    sal_val = []
    for i in range(len(df)):
        if df.salary[i] == ' <=50K':
            sal_val.append(0)
        else:
            sal_val.append(1)

    df['salary'] = sal_val
    return df


try:
    df=pd.read_csv('adult.csv')
except Exception as e:
    print("some thing went wrong during reading adult.csv file")
else:
    print(df.head())
    print(df.shape)

    #Create profile report
    # getprofilereport(df,'Adult_report1.html')



    #peroform convert salary to num
    df=cattonum(df)

    # #peroform onehat encoding for the categorical variables
    catcols,df=onehat(df)
    getprofilereport(df, 'Adult_report2.html')

    X = df
    X.drop('salary', axis=1)
    y=df['salary']
    #
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=30)
    #
    alg_SGD_clf = SGDClassifier(random_state=52)
    alg_SGD_clf.fit(x_train, y_train)
    y_pred=alg_SGD_clf.predict(x_test)
    num_correct=sum(y_pred==y_test)
    accuracy = num_correct / len(y_test)
    print(accuracy)

    # using cross_val_score() function

    from sklearn.model_selection import cross_val_score

    print(cross_val_score(alg_SGD_clf, x_train, y_train, cv=3, scoring="accuracy"))
    # Accuracy is generally not the preferred perfomance measure for classifiers, especially when you are dealing
    # with skewed data set(i.e. wehn some classes are much more frequent than others)
    # Let us look at the dumb classifer that just classifes every single image in the not-5 class i.e it classifies each
    # and every instance as not-5 only i.e false result only for all the instances.
    # With this dummy false classifier also the accuracy is above 90% hence accuracy is not a good measure for classifiers.
    # For the output, you'll get no less than 90%: only 10% of the images are 5s, so we
    # can always imagine that an image is not a 5. We'll be right about 90% of the time.
    from sklearn.base import BaseEstimator
    import numpy as np


    class Neveryesclassifier(BaseEstimator):

        def fit(self, X, y=None):
            pass

        def predict(self, X):
            return np.zeros((len(X), 1),
                            dtype=bool)  # just returns the zero in boolian i.e. False for the length of X values


    never_yes_clf = Neveryesclassifier()
    print(cross_val_score(never_yes_clf, x_train, y_train, cv=3, scoring="accuracy"))

    # Confusion Matrix
    # There is a better method to evaluate the performance of your classifier: the  confusion matrix.

    from sklearn.model_selection import cross_val_predict

    y_train_pred = cross_val_predict(alg_SGD_clf, x_train, y_train, cv=3)

    print(y_train_pred)
    from sklearn.metrics import confusion_matrix

    confusion_matrix(y_train, y_train_pred)

    # Precision Recall calculations
    from sklearn.metrics import precision_score, recall_score

    print(f"Precision = {precision_score(y_train, y_train_pred)}")
    print(f"Recall = {recall_score(y_train, y_train_pred)}")

    # F1 score calculation
    from sklearn.metrics import f1_score

    print(f1_score(y_train, y_train_pred))

    # using cross_val_prediction() with specifying decision_function instead of scores
    # Decision_function
    y_scores = cross_val_predict(alg_SGD_clf, x_train, y_train, cv=3, method='decision_function')
    print(y_scores)  # contains the scores only

    # Precision_recall curve
    from sklearn.metrics import precision_recall_curve

    precision,recall,thresholds=precision_recall_curve(y_train,y_scores)

    # Precision Recall Threshold trade off curve
    import matplotlib.pyplot as plt


    def plot_precision_recall_vs_threshold(precision, recalls, thresholds):
        plt.plot(thresholds, precision[:-1], "b-", label="Precision")
        # here precision[:-1] is given to omit the last precision
        # value to match the length of the thresholds. The length of
        # threshold is 1 less than precision and recall
        plt.plot(thresholds, recall[:-1], 'g-', label="recall")
        plt.legend(('Precision', 'Recall'))
        plt.xlabel('Threshold')
        plt.ylabel('Prcision And Recall')

        plt.grid(color='r', linestyle='-', linewidth=2)


    plot_precision_recall_vs_threshold(precision, recall, thresholds)
    plt.show()

    # Precision Versus Recall curve
    # Another way to select a good precsion/recall trade-off is to plot precison directly against recall.

    import matplotlib.pyplot as plt


    def plot_precision_recall_vs_threshold(precision, recalls, thresholds):
        plt.plot(recall, precision, "b-", label="Precision")
        # here precision[:-1] is given to omit the last precision
        # value to match the length of the thresholds. The length of
        # threshold is 1 less than precision and recall

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        #   plt.xticks([.8])
        #   plt.yticks([0.74])
        plt.grid(color='r', linestyle='-', linewidth=2)


    plot_precision_recall_vs_threshold(precision, recall, thresholds)
    plt.show()

    # ROC curve (Reciever operating characteristic)
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_train, y_scores)  # Technically roc_curve is a function to which you are passing


    # the variables as y_train and y_score,
    # then the function is returning FPR, TPR and Thresholds.


    def plot_roc_curve(fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')  # dashed diagonal
        plt.plot([0], [.59], '--')
        plt.xlabel("False Positive Rate FTR ")
        plt.ylabel("True Positive Rate TPR or Recall")
        plt.yticks(ticks=[.59])
        plt.annotate("Chosen Ratio", [0, .59])
        plt.annotate("ROC curve", [.55, .55])
        plt.grid()


    plot_roc_curve(fpr, tpr)
    plt.show()

    # One way to compare the classifiers is to measure the area under the curve(AUC).
    # A perfect classifier will have AUC of ROC as 1
    # Purely random classifer will have ROC AUC as .5.
    # AUC of ROC can be computed using roc_auc_score function
    from sklearn.metrics import roc_auc_score

    roc_auc_score(y_train, y_scores)

    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    alg_randomforest_clf = RandomForestClassifier(random_state=52)
    alg_randomforest_clf.fit(x_train, y_train)
    y_pred_forest = alg_randomforest_clf.predict(x_test)
    num_correct_forrest = sum(y_pred_forest == y_test)
    print(num_correct_forrest)
    accuracy = num_correct_forrest / len(y_pred_forest)
    print(accuracy)

    # cross validation randomforest
    from sklearn.model_selection import cross_val_score

    print(cross_val_score(alg_randomforest_clf, x_train, y_train, cv=3, scoring='accuracy'))

    # Confusion Metrix
    from sklearn.model_selection import cross_val_predict

    y_cross_pred_forest = cross_val_predict(alg_randomforest_clf, x_train, y_train, cv=3)
    print(confusion_matrix(y_train, y_cross_pred_forest))
    print(confusion_matrix(y_test, y_pred_forest))
    precision = precision_score(y_test, y_pred_forest)
    print(precision)
    recall = recall_score(y_test, y_pred_forest)
    print(recall)
    f1_score(y_test, y_pred_forest)

    # Random forest does not have decision function instead it has predict_proba() function
    y_forest_score = cross_val_predict(alg_randomforest_clf, x_train, y_train, cv=3, method='predict_proba')

    y_forest_single_score = y_forest_score[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_forest_single_score)


    def plot_precision_recall_vs_thresholds(precisions, recall, threshold):
        plt.plot(threshold, precisions[:-1], 'b--')
        plt.plot(threshold, recall[:-1], 'g-')
        plt.xlabel("Threshold")
        plt.ylabel("Precision and Recall")
        plt.legend(["Preciosn", "Recall"])
        plt.grid()


    plot_precision_recall_vs_thresholds(precisions, recalls, thresholds)
    plt.show()


    def plot_precision_recall_vs_thresholds(precisions, recall, threshold):
        plt.plot(recalls, precisions, 'b-')

        plt.xlabel("Recall")
        plt.ylabel("Precisions")

        plt.grid()


    plot_precision_recall_vs_thresholds(precisions, recalls, thresholds)
    plt.show()

    fpr_forest,tpr_forest,tthresholds_forest=roc_curve(y_train,y_forest_single_score)
    plt.plot(fpr, tpr, 'b:')
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    plt.legend(["SGD", "Random Forest"], loc='lower right')
    plt.show()

    # Conclusion
    print("""As you can see the RandomForestClassifier's ROC curvelooks much better than the SGD classifier's. It comes
    much closer to the top-left corner. As a result, its ROC AUC score is also significantly better. And by measuring the 
    precision recall score, we found 100% of precision with recall at 100%. So the RandomForestClassifier has improved
    model performance as compared to SGDClassifier model""")













