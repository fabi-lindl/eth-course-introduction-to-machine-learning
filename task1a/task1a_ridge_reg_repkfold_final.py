import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, RepeatedKFold

# Parameters
lambdas = [0.1, 1, 10, 100, 200]
num_folds = 10
fit_intercept = False
ridge_normalize = False
ridge_solver = 'auto' # svd, cholesky, lsqr, sparse_cg, sag, saga
random_state = 42 # for test/train split function
shuffle = True # shuffle data before splitting

n_repeats = 1000
n_splits = num_folds

df_train = pd.read_csv('./dataset/train.csv')

# Define feature matrix X and lables y
X = df_train.drop('y', axis=1)
y = df_train['y']

# Try different regularization params and save results to a file
results_path = (f'./results/submission_repeatedKfold_{n_repeats}_'
                f'intercept_{fit_intercept}_random_{random_state}.csv')
with open(results_path, 'w') as file:
    scoring = 'neg_root_mean_squared_error'
    for lambda_ in lambdas:
        model = Ridge(alpha=lambda_,
                      fit_intercept=fit_intercept,
                      normalize=ridge_normalize,
                      solver=ridge_solver,
                      random_state=random_state)
        rkf = RepeatedKFold(n_splits=n_splits,
                            n_repeats=n_repeats,
                            random_state=random_state)
        rmse_cv = cross_val_score(model, X, y, scoring=scoring, cv=rkf)
        rmse_cv = abs(rmse_cv)
        print(f'lambda {lambda_}: rmse mean = {rmse_cv.mean()}')
        rmse_mean = '{:.12f}'.format(rmse_cv.mean())
        file.write(rmse_mean+'\n')
        del model
