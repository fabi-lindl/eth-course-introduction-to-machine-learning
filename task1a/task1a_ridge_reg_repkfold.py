import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, RepeatedKFold

# Read in data. 
labels, features = np.zeros(150), np.zeros(150*13).reshape(150, 13)
with open('train.csv', newline='') as csvfile:
    tdr = csv.reader(csvfile, delimiter=',')
    next(tdr)
    cnt = 0
    for sample in tdr:
        labels[cnt] = sample[0]
        features[cnt] = sample[1:]
        cnt+=1

# Compute the ridge regression. 

# Parameters. 
nfolds = 10 # No. of folds for crossval. 
nrepeats = 100 # No. of repetitions for repeated k-fold. 
lambdas = [0.1, 1, 10, 100, 200] # Hyperparameters to choose. 
rmses = [] # Computed RMSEs

for i in lambdas:
    ridge = Ridge(alpha=i, fit_intercept=False, solver='auto')
    rkfold = RepeatedKFold(n_splits=nfolds, n_repeats=nrepeats, random_state=42)
    scores = cross_val_score(ridge, features, labels, cv=rkfold, scoring='neg_root_mean_squared_error')
    meanScore = np.mean(scores)
    rmses.append(abs(meanScore))

print(rmses)

# Save data. 
with open('ridge_reg_repkfold.csv', mode='w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(5):
        csvwriter.writerow([rmses[i]])
