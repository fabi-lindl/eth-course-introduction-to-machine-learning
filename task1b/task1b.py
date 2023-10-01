import csv
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split

def create_feature_vector(sample):
    """
    sample: Numpy array [1x5]
    Returns a numpy array [1x21]
    -----
    Maps the raw sample data to a feature vector of provided X.
    """
    fvector = np.zeros(21)
    lim = len(sample)
    cnt = 0
    for i in range(5):
        cnt = i*5
        if i == 0:
            for k in range(5):
                fvector[k+cnt] = sample[k]
        elif i == 1:
            for k in range(5):
                s = float(sample[k])
                fvector[k+cnt] = s*s
        elif i == 2:
            for k in range(5):
                fvector[k+cnt] = np.exp(float(sample[k]))
        elif i == 3:
            for k in range(5):
                fvector[k+cnt] = np.cos(float(sample[k]))
        else:
            fvector[cnt] = 1
    return fvector

# Read in data
y, X = np.zeros(700), np.zeros(700*21).reshape(700, 21)
with open('./dataset/train.csv', newline='') as csvfile:
    tdr = csv.reader(csvfile, delimiter=',')
    next(tdr) # Skip header.
    cnt = 0
    for sample in tdr:
        y[cnt] = float(sample[1])
        X[cnt] = create_feature_vector(sample[2:])
        cnt+=1

# Compute rmse using Ridge regression

# Parameters.
nfolds = 8 # No. of folds for crossval
nrepeats = 4 # No. of repetitions for repeated k-fold
lambdas = [0.1, 1, 10, 100, 200, 400]
rmses = [] # Computed RMSEs

for i in lambdas:
    ridge = Ridge(alpha=i, fit_intercept=False, solver='auto')
    rkfold = RepeatedKFold(n_splits=nfolds, n_repeats=nrepeats,
                           random_state=None)
    scores = cross_val_score(ridge, X, y, cv=rkfold,
                             scoring='neg_root_mean_squared_error')
    meanScore = np.mean(scores)
    rmses.append(abs(meanScore))

print(rmses)

# Get lambda with smallest rmse.
minVal = min(rmses)
lambda_ = lambdas[rmses.index(minVal)]
print(f'lambda = {lambda_}')

# Compute the weights. 
# Split the data set into training data and test data. 
trainSize = 0.75 # Size of the training data in percent. 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainSize,
                                                    random_state=None)
# Make a prediction on the split test set.
model = Ridge(alpha=lambda_, fit_intercept=False, solver='auto')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
rmseModel = np.sqrt(mean_squared_error(y_test, y_predict))
weights = model.coef_

print(f'Model RMSE = {rmseModel}')
print('---------- Weights ----------')
for i in weights:
    print(i)

# Store results to file.
with open('./results/task1b_solution.csv', mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    for i in range(21):
        csvwriter.writerow([weights[i]])
