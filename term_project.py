"""
------------------------------------------------------------------------
Author: Austin Bursey
ID:     160165200
Email:  burs5200@mylaurier.ca
__updated__ = "2019-09-23"
------------------------------------------------------------------------
"""

import pandas 
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.model_selection import GridSearchCV
import numpy as np
from math import sqrt


def VisualizeDataset(dataset): 
    '''
    Visualize
    And
    Summarize
    The Dataset
    '''
    #summarize the data set 
    # shape
    print(dataset.shape)

    print(dataset.head(20))
    # descriptions
    print(dataset.describe())

    print("----------------------------------")
    #class distribution
    print(dataset.groupby('Moves till Checkmate').size())

    #box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(3,2), sharex=False, sharey=False)
    plt.show()
    dataset.hist()
    plt.show()

    #Correlation Matrix
    pandas.plotting.scatter_matrix(dataset,  figsize=(6, 6))
    plt.show()

def plotOptimalK(desc_train,targ_train,splits):
    scoring = 'accuracy'
    i_array = list()
    euclid = list()
    euclid_W = list()
    manhattan = list()
    manhattan_W = list()
    Gaussian = list()
    for i in range(1,50):
        models =[]
        models.append(('KNN-Euclid', KNeighborsClassifier(n_neighbors=i, p=2)))
        models.append(("KNN-Euclid-Weighted", KNeighborsClassifier(n_neighbors=i,weights='distance',p=2 )))
        models.append(('KNN-Manhattan', KNeighborsClassifier(n_neighbors=i, p=1)))
        models.append(("KNN-Manhattan-Weighted", KNeighborsClassifier(n_neighbors=i,weights='distance',p=1 )))    
        #models.append(("Gaussian Bayes", GaussianNB()))

        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            models
            kfold = model_selection.StratifiedKFold(n_splits=splits,  random_state=seed)
            cv_results = model_selection.cross_val_score(model, desc_train, targ_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            # print(msg)
            # print("-")
            if name == 'KNN-Euclid': 
                euclid.append( cv_results.mean())
            elif name == "KNN-Euclid-Weighted": 
                euclid_W.append( cv_results.mean())
            elif name == 'KNN-Manhattan': 
                manhattan.append( cv_results.mean())
            elif name == 'KNN-Manhattan-Weighted': 
                manhattan_W.append( cv_results.mean())
            # elif name == 'Gaussian Bayes': 
            #     Gaussian.append( cv_results.mean())
        
        i_array.append(i)

    plt.plot(i_array,euclid, label= 'Euclidian')
    plt.plot(i_array,euclid_W, label= 'Euclidian Weighted')
    plt.plot(i_array,manhattan, label= 'Manhattan ')
    plt.plot(i_array,manhattan_W, label= 'Manhattan Weighted')

    plt.legend( loc='upper left')
    plt.title("Accuracy per (K)")
    plt.xlabel("K values")
    plt.ylabel("Accuracy")
    plt.show()

if __name__ == "__main__": 
    '''
    Variables 
    To 
    Change 
    Between
    Runs
    '''
    validation_size = 0.1
    seed = 58
    neighbhors = 8
    splits =100


    # Load dataset
    url = "./krkopt.csv"
    names = ["White-King-Col", "White-King-Row", "White-Rook-Col", "White-Rook-Row","Black-King-Col", "Black-King-Row","Moves till Checkmate" ]
    dataset = pandas.read_csv(url, names=names)
    array = dataset.values
    
    descriptiveFeats = array[:,0:6]
    targetFeats = array[:,6]

    sm = SMOTE(random_state=seed)
    descriptiveFeats, targetFeats =sm.fit_resample(descriptiveFeats,targetFeats)
    '''
    Visualize
    And
    Summarize
    The Dataset
    '''
    VisualizeDataset(dataset)

    scoring = 'accuracy'
    #desc_train, desc_validation, targ_train, targ_validation = model_selection.train_test_split(descriptiveFeats, targetFeats, test_size=validation_size, random_state=seed,stratify=targetFeats)
    
    '''
    Find
    Optimal 
    K
    '''
    #findOptimalK(desc_train,targ_train)
    
    
    
    models =[]
    models.append(('KNN-Euclid', KNeighborsClassifier(n_neighbors=neighbhors, p=2)))
    models.append(("KNN-Euclid-Weighted", KNeighborsClassifier(n_neighbors=neighbhors,weights='distance',p=2 )))
    models.append(('KNN-Manhattan', KNeighborsClassifier(n_neighbors=neighbhors, p=1)))
    models.append(("KNN-Manhattan-Weighted", KNeighborsClassifier(n_neighbors=neighbhors,weights='distance',p=1 )))    
    models.append(("Gaussian Bayes", GaussianNB()))

    #evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        cv_results = model_selection.cross_val_score(model, descriptiveFeats, targetFeats, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


    # for name, model in models: 
    #     print("-----{}------".format(name))
    #     model.fit(desc_train,targ_train)
    #     predictions = model.predict(desc_validation)
    #     print(accuracy_score(targ_validation, predictions))
    #     print(confusion_matrix(targ_validation, predictions))
    #     print(classification_report(targ_validation, predictions))
    #     print()