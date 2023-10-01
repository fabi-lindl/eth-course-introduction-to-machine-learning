import glob
import numpy as np
import operator
import os
import pandas as pd
import pickle

from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from generate_features import *
from parameters import *

def clear_features_directory(clean):
    """
    Deletes feature file if clean == True. 
    """
    if clean:
        files = glob.glob('./data/features/*_features.csv')
        for f in files:
            try:
                os.remove(f)
                print(f'Deleted feature file {f}.')
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

def run_grid_search(results, model_params, cv, X_train, y_train):
    """
    Performs a grid search on the given model and parameter set. 
    Stores results in the provided argument results. 
    ---------
    results: List to store grid search results to. 
    model_params: Dict with model parameters.
    cv: Cross validation object. 
    """
    print('Run grid search ...') 
    for params in model_params:
        clf = params['clf'][0]
        print(f'Fitting classifier {clf}')
        
        # Pop classifier from model parameters, as it is no parameter. 
        params.pop('clf')

        steps = [('scaling', StandardScaler()), ('clf', clf)]
        grid = GridSearchCV(Pipeline(steps), param_grid=params, cv=cv,
                            scoring='f1', n_jobs=-1, verbose=5)
        grid.fit(X_train, y_train)

        results.append(
            {
                'grid': grid,
                'classifier': grid.best_estimator_,
                'best_score': grid.best_score_,
                'best_params': grid.best_params_,
            }
        )

def generate_pickle_file_with_best_model_parameters(
    model_params, cv, X_train, y_train):
    results = []
    run_grid_search(results, model_params, cv, X_train, y_train)
    results = sorted(results, key=operator.itemgetter('best_score'), reverse=True)
    
    best_grid = results[0]['grid']
    print(f'Best grid {best_grid}')

    pickle_fname = 'grid_search_model_and_parameters.p'
    pickle.dump(results, open('./data/'+pickle_fname, 'wb'))
    return results

def show_best_gs_model(results, X_train, y_train, X_valid, y_valid):
    """
    Prints the best gs performing model and its parameters.
    Runs a validation on the validation set and prints the score. 
    """
    # Result at position zero is the best one. 
    best_clf = results[0]['classifier'][1]
    best_params = results[0]['best_params']
    best_score = results[0]['best_score']

    print(f'CV best clf {best_clf}')
    print(f'CV best score for {best_score}')
    print(f'Best parameters {best_params}')

    # Run validation with the best classifier and parameters. 
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_valid)
    f1 = f1_score(y_true=y_valid, y_pred=y_pred)
    print(f'Validation f1 score after refitting = {f1}')

def create_submission_file_with_best_gs_model(
    path_submission, results,X_train, y_train, X_test):
    """
    Creates a .csv file with the predictions on the provided test set using 
    the best performing model yielded by grid search. 
    results: List of grid searched models with their best parameter sets. 
    """
    best_clf = results[0]['classifier'][1]
    best_clf_name = best_clf.__class__.__name__
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    df_res = pd.DataFrame(y_pred.T)
    gs_submission_filename = f'gs_best_prediction_submission_{best_clf_name}.csv'
    df_res.to_csv(path_submission+gs_submission_filename, index=False, header=None)

def predict_test_labels(
    path_submission, X_train, y_train, X_valid, y_valid, X, y, X_test):
    """
    Reads in data from a pickle file that was created via grid search.
    Predicts test labels for the provided test set for each classifier
    stored in the pickle file. 
    Stores the predictions as .csv file in './data/submit/'. 
    """
    # Read best parameters of trained models.
    path = path_submission + 'submission_from_pickle_file_' 
    grid_search_results = pd.read_pickle(r'./data/grid_search_model_and_parameters.p')
    for i in grid_search_results:
        clf = i['classifier'][1]
        clf_name = clf.__class__.__name__

        # Evaluate model and print user information. 
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)
        f1 = f1_score(y_valid, y_pred)
        print(f'{clf_name}: Validation f1 score = {f1}')

        # Fit classifier on the whole data and predict labels for the test set. 
        clf.fit(X, y)
        y_pred = clf.predict(X_test)
        
        # Store predictions in submission folder. 
        fpath = path + clf_name + '.csv'
        df_res = pd.DataFrame(y_pred)
        df_res.to_csv(fpath, index=False, header=None)

if '__main__' == __name__:
    clear_features_directory(clean)

    # load data. 
    if test_pipeline: # use simulated dataset
        feature_file_train = './data/features/train_sim_features.csv'
        df_train = pd.read_csv('./data/train_sim.csv')    
        df_test = pd.read_csv('./data/test_sim.csv')
        if submit:
            feature_file_test = './data/features/test_sim_features.csv'
    else:
        feature_file_train = './data/features/train_features.csv'
        df_train = pd.read_csv('./data/train.csv')
        df_test = pd.read_csv('./data/test.csv')
        if submit:
            feature_file_test = './data/features/test_features.csv'

    # Generate features
    if os.path.isfile(feature_file_train):
        print ("Feature file for training data exists")
        df_train = pd.read_csv(feature_file_train)
    else:
        print ("Features for training set are being created...")
        generate_features_onehot(df_train, same_aminos=same_aminos,
                                 path_to_save=feature_file_train)
        # Check if file was created. 
        assert os.path.isfile(feature_file_train)
        df_train = pd.read_csv(feature_file_train)

    X = df_train.drop(['Active', 'Sequence'], axis=1)
    y = df_train['Active']

    if submit:
        if os.path.isfile(feature_file_test):
            print ("Feature file for test data exists")
            df_test = pd.read_csv(feature_file_test)
        else:
            print ("Features for test set are being created...")
            generate_features_onehot(df_test, same_aminos=same_aminos,
                                     path_to_save=feature_file_test)
            assert os.path.isfile(feature_file_test)
            df_test = pd.read_csv(feature_file_test)

        X_test = df_test.drop(['Sequence'], axis=1)
        df_test = df_test.set_index('Sequence')

        # Order of columns matter, as pd will convert to series.values and
        # only consider numbers!
        X_test = X_test[X.columns]
        assert np.array_equal(X_test.columns, X.columns) 

    # Create train test split. 
    if stratify:
        stratify = y
    else:
        stratify = None
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=train_size,
                                                          stratify=stratify)

    # Standardize data in case grid search is not run. 
    # Grid search implements data standardization by default. 
    if scaling and not grid_cv:
        print('Data is scaled hardcoded.')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        if submit:
            # Scale the whole data set for the submission prediction.
            X = scaler.fit_transform(X)
            X_test = scaler.transform(X_test)
    else:
        print('Data is not scaled hardcoded.')

    if grid_cv:
        # Perform grid search to find the best parameters for the provided models. 
        results = generate_pickle_file_with_best_model_parameters(model_params, cv, X_train, y_train)
        show_best_gs_model(results, X_train, y_train, X_valid, y_valid)
        if submit:
            create_submission_file_with_best_gs_model(path_submission, results, X, y, X_test)
            predict_test_labels(path_submission, X_train, y_train, X_valid, y_valid, X, y, X_test)
    else:
        predict_test_labels(path_submission, X_train, y_train, X_valid, y_valid, X, y, X_test)
