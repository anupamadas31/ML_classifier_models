from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
np.set_printoptions(threshold=np.inf)
import numpy as np
from sklearn.metrics import accuracy_score


df=pd.read_csv(r'path/to/csvfolder/input.csv')



X = df.iloc[:,[1,2,3,4,5,6,7,8,9,11,12] ].values #feature columns
# print(X)
Y = df.iloc[:, 13].values #target feature
# print(Y)
estimators = []
estimators.append(('standardize', preprocessing.Normalizer()))#normalize the values as it affects the model
model = MLPClassifier(solver='lbfgs',activation='relu', alpha=10, hidden_layer_sizes=(30,2), random_state=1)
estimators.append(('mlp', model))
pipeline = Pipeline(estimators)

seed = 7
np.random.seed(seed)
kfold = KFold(n_splits=33, random_state=seed)
Y_Predicted = cross_val_predict(pipeline, X, Y, cv=kfold)

model.fit(X,Y)
pred_prob=model.predict_proba(X)
# print(pred_prob)


accuracy= accuracy_score(Y, Y_Predicted) * 100
confusionMatrix= confusion_matrix(Y, Y_Predicted)
print(confusionMatrix)
print('accuracy is',accuracy)

