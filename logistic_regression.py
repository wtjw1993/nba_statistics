# logistic regression
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
np.random.seed(100)

# read training data
train_allNBA = pd.read_csv('train_allNBA.csv').fillna(0)
predictors = [c for c in train_allNBA.columns if "allNBA" not in c]

# test data from 2016-17 NBA season
# downloaded from https://bkref.com/pi/shareit/GA6XJ and modified
test2016 = pd.read_csv('nba2016_17.csv').fillna(0)
test2016 = pd.concat((test2016[['Player','playerID','pos']], pd.get_dummies(test2016['pos'],prefix='pos',drop_first=True), test2016.iloc[:,3:]), axis=1)
# test data from 2015-16 NBA season
# downloaded https://bkref.com/pi/shareit/hLxCY and modified
test2015 = pd.read_csv('nba2015_16.csv').fillna(0)
test2015 = pd.concat((test2015[['Player','playerID','pos']], pd.get_dummies(test2015['pos'],prefix='pos',drop_first=True), test2015.iloc[:,3:]), axis=1)

# Standardise predictors
scaler = StandardScaler()
scaler.fit(train_allNBA[predictors[2:]])
X_train = pd.DataFrame(scaler.transform(train_allNBA[predictors[2:]]))
X_train.columns = predictors[2:]
X_train = pd.concat((train_allNBA[predictors[:2]], X_train), axis = 1)
X_test2016 = pd.DataFrame(scaler.transform(test2016[predictors[2:]]))
X_test2016.columns = predictors[2:]
X_test2016 = pd.concat((test2016[predictors[:2]], X_test2016), axis = 1)
X_test2015 = pd.DataFrame(scaler.transform(test2015[predictors[2:]]))
X_test2015.columns = predictors[2:]
X_test2015 = pd.concat((test2015[predictors[:2]], X_test2015), axis = 1)

# Split training set into a training and a calibration set
X_train, X_cal, y_train, y_cal= train_test_split(X_train, train_allNBA['allNBA'], test_size = 0.5)

# Use GridSearchCV to determine best parameters for logitReg
t0 = time.time()
logit = LogisticRegression()
parameters = {'C': [0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2'], 'class_weight': [None, 'balanced']}
stratKFolds = StratifiedKFold(n_splits = 3)
logitGrid = GridSearchCV(logit, parameters, scoring = 'f1', cv = stratKFolds)
logitGrid.fit(X_train, y_train)
t1 = time.time()
print(logitGrid.best_params_)
print("Time taken:", str(t1-t0), "seconds")

# Use StratifiedKFold and CalibratedClassifierCV for logitReg
t0 = time.time()
logit = logitGrid.best_estimator_
logitCV = CalibratedClassifierCV(logit, cv = stratKFolds)
logitCV.fit(X_cal, y_cal)
t1 = time.time()
print("Time taken:", str(t1-t0), "seconds")

# NBA 2016-17 All-NBA Team Predictions
probs2016 = pd.DataFrame(logitCV.predict_proba(X_test2016))
probs2016 = pd.concat((test2016[['Player','pos']], probs2016), axis = 1)
pred2016 = probs2016.sort_values(1, ascending = False).iloc[:15,:] # select only the first 15 observations with the highest probability of being classified positive
print("\nPredicted 2016-17 All-NBA Team winners:\n")
print(pred2016[['Player','pos']].sort_index())
print("\nActual 2016-17 All-NBA Team winners:\n")
print(test2016.loc[test2016['allNBA'] == 1, ['Player','pos']])
print("\nNumber of correct All-NBA predictions:", len(set(pred2016['Player']) & set(test2016.loc[test2016['allNBA'] == 1, 'Player'])))
print("\nDecision threshold:", str(pred2016.iloc[14,3].round(3)))
probs2016['Label'] = 0
for i, label in enumerate(probs2016[1]):
    if label >= pred2016.iloc[14,3]:
        probs2016['Label'].iloc[i] = 1
print("\nClassifier F1 score:", f1_score(test2016['allNBA'], probs2016['Label']).round(3))

# NBA 2015-16 All-NBA Team Predictions
probs2015 = pd.DataFrame(logitCV.predict_proba(X_test2015))
probs2015 = pd.concat((test2015[['Player','pos']], probs2015), axis = 1)
pred2015 = probs2015.sort_values(1, ascending = False).iloc[:15,:] # select only the first 15 observations with the highest probability of being classified positive
print("\nPredicted 2015-16 All-NBA Team winners:\n")
print(pred2015[['Player','pos']].sort_index())
print("\nActual 2015-16 All-NBA Team winners:\n")
print(test2015.loc[test2015['allNBA'] == 1, ['Player','pos']])
print("\nNumber of correct All-NBA predictions:", len(set(pred2015['Player']) & set(test2015.loc[test2015['allNBA'] == 1, 'Player'])))
print("\nDecision threshold:", str(pred2015.iloc[14,3].round(3)))
probs2015['Label'] = 0
for i, label in enumerate(probs2015[1]):
    if label >= pred2015.iloc[14,3]:
        probs2015['Label'].iloc[i] = 1
print("\nClassifier F1 score:", f1_score(test2015['allNBA'], probs2015['Label']).round(3))
