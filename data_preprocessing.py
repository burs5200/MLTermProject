import pandas 
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
import numpy as np

url = "./krkopt.data"
names = ["White-King-Col", "White-King-Row", "White-Rook-Col", "White-Rook-Row","Black-King-Col", "Black-King-Row","Moves till Checkmate" ]
dataset = pandas.read_csv(url, names=names)
array = dataset.values
for row in array: 
    for i in range(len(row)): 
        if row[i] == 'a' : 
            row[i] = 1 
        elif row[i] == 'b': 
            row[i] = 2 
        elif row[i] == 'c': 
            row[i] = 3 
        elif row[i] == 'd': 
            row[i] = 4 
        elif row[i] == 'e': 
            row[i] = 5 
        elif row[i] == 'f': 
            row[i] = 6 
        elif row[i] == 'g': 
            row[i] = 7 
        elif row[i] == 'h': 
            row[i] = 8                                                        

np.savetxt("krkopt.csv",array, delimiter=',',fmt='%s')