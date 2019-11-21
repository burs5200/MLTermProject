"""
------------------------------------------------------------------------
Author: Austin Bursey
ID:     160165200
Email:  burs5200@mylaurier.ca
__updated__ = "2019-09-23"
------------------------------------------------------------------------
"""
# i need these libraries
import pandas 
import matplotlib.pyplot as plt
from sklearn import model_selection


#i think i need these libraries 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import ADASYN, SMOTE
#why are they here 

from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import seaborn as sns
from math import sqrt

def chessMoves(x, y ):
    movesAway= 0 
    #kings use euclidian
    movesAway+= sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[4]-y[4])**2 + (x[5]-y[5])**2)
    if (x[3] != y[3]): 
        movesAway +=1 
    if (x[4] != y[4]): 
        movesAway +=1 
    return movesAway


if __name__ == "__main__": 
    '''
    Variables 
    To 
    Change 
    Between
    Runs
    '''
    validation_size = 0.001
    seed = 58
    neighbhors = 10
    splits =15


    # Load dataset
    url = "./krkopt.csv"
    names = ["White-King-Col", "White-King-Row", "White-Rook-Col", "White-Rook-Row","Black-King-Col", "Black-King-Row","Moves till Checkmate" ]
    dataset = pandas.read_csv(url, names=names)
    array = dataset.values
    
    descriptiveFeats = array[:,0:6]
    targetFeats = array[:,6]

    sm = SMOTE(random_state=seed)
    descriptiveFeats, targetFeats =sm.fit_resample(descriptiveFeats,targetFeats)

    #ada = ADASYN(sampling_strategy= 'all', random_state=seed,n_neighbors=neighbhors,ratio='all')
    #descriptiveFeats, targetFeats = ada.fit_resample( descriptiveFeats ,targetFeats )
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
    # class distribution
    print(dataset.groupby('Moves till Checkmate').size())

    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(3,2), sharex=False, sharey=False)
    plt.show()
    dataset.hist()
    plt.show()



    desc_train, desc_validation, targ_train, targ_validation = model_selection.train_test_split(descriptiveFeats, targetFeats, test_size=validation_size, random_state=seed,stratify=targetFeats)

    scoring = 'accuracy'

    models =[]
    models.append(('KNN-Euclid', KNeighborsClassifier(n_neighbors=neighbhors, p=2)))
    models.append(("KNN-Euclid-Weighted", KNeighborsClassifier(n_neighbors=neighbhors,weights='distance',p=2 )))
    models.append(('KNN-Manhattan', KNeighborsClassifier(n_neighbors=neighbhors, p=1)))
    models.append(("KNN-Manhattan-Weighted", KNeighborsClassifier(n_neighbors=neighbhors,weights='distance',p=1 )))
    models.append(('KNN-Custom Distance', KNeighborsClassifier(n_neighbors=neighbhors, p=1,metric=chessMoves)))
    models.append(("KNN-Custom-Weighted", KNeighborsClassifier(n_neighbors=neighbhors,weights='distance', p=1 ,metric=chessMoves)))
        
    models.append(("Gaussian Bayes", GaussianNB()))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        models
        kfold = model_selection.StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        cv_results = model_selection.cross_val_score(model, desc_train, targ_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


    for name, model in models: 
        print("-----{}------".format(name))
        model.fit(desc_train,targ_train)
        predictions = model.predict(desc_validation)
        print(accuracy_score(targ_validation, predictions))
        print(confusion_matrix(targ_validation, predictions))
        print(classification_report(targ_validation, predictions))
        print()