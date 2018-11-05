from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
np.set_printoptions(threshold=np.inf)
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm
import collections
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

df =pd.read_csv(r'D:/stitched carinata/1row/csv_new/upsample/upsampled_data.csv')#load dataset

X= df.iloc[:,[1,2,3,5,7,8,9,11] ].values #feature columns
Y=df.iloc[:,13].values#target values

classify = Pipeline(((("scaler", StandardScaler()),("svm_clf",svm.SVC(kernel="poly",degree=6,gamma=2, C=0.1)))))#svm model

seed = 7
np.random.seed(seed)
kfold = KFold(n_splits=10,random_state=seed)
Y_Predicted = cross_val_predict(classify, X, Y, cv=kfold)
# prediction = pd.DataFrame(Y_Predicted, columns=['predictions']).to_csv('prediction_svm.csv')
classify.fit(X,Y)
score_folds=cross_val_score(classify, X,Y,cv=10)
print(score_folds)
acc=(accuracy_score(Y, Y_Predicted) * 100)
print(acc)
confusionMatrix = confusion_matrix(Y, Y_Predicted)
print(confusionMatrix)


# ############plotting the confusionn matrix#############
def plot_confusion_matrix(cm, target_names, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title('Random Forest,kfold=10',size=15)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape
    fmt =  'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",size= 20,style='oblique')

    plt.ylabel('True label',size=15)
    plt.xlabel('Predicted label',size=15)


plt.figure()
plot_confusion_matrix(confusionMatrix, target_names=['1','2','3'])
plt.show()

