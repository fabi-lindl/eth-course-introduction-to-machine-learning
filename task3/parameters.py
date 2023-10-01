""" All parameters for model training and testing are defined in this file. """

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

# Submission and prediction details.
submit = True
test_pipeline = False # For testing different features.
path_submission = './data/submit/'

# Feature encoding
clean = True # clean feature directory (create new feature files)
same_aminos = True

# Cross validation. 
train_size = 0.8
stratify = True

# Gridsearch. 
grid_cv = True
cv =  StratifiedKFold(n_splits=5, shuffle=True) # grid_cv = True
scaling = True # grid_cv = False

# Models. 
model_params = [
    # {
    #     'clf': [RandomForestClassifier()],
    #     'clf__n_estimators': [100, 200],
    #     'clf__max_depth': [None, 2, 5],
    #     'clf__n_jobs': [-1],
    #     'clf__class_weight': [None, 'balanced']
    # },
    # {
    #     'clf': [HistGradientBoostingClassifier()], 
    #     'clf__learning_rate': [1],
    #     'clf__max_leaf_nodes':[50, 100, 200, 400, 700, 1000],
    #     'clf__l2_regularization':[0.01, 0.1, 1, 5, 10, 20, 100],
    # },


    # {
    #     'clf': [SVC()],
    #     'clf__C' :[2**2, 2**4, 2**6, 2**10],
    #     'clf__gamma':['scale', 'auto', 10**-4, 10**-1],
    #     'clf__kernel':['rbf'],
    #     'clf__class_weight': ['balanced']
    # },
    # Best RandomForest
    {
        'clf': [RandomForestClassifier()],
        'clf__n_estimators': [200],
        'clf__max_depth': [None],
        'clf__n_jobs': [-1],
        'clf__class_weight': ['balanced']

    },
    # Best HistGradBoost. 
    {
        'clf': [HistGradientBoostingClassifier()], 
        'clf__learning_rate': [1],
        'clf__max_leaf_nodes':[200],
        'clf__l2_regularization':[100],
    },
    # Best SVC
    {
        'clf': [SVC()],
        'clf__C' :[2**4],
        'clf__gamma':['scale'],
        'clf__kernel':['rbf'],
        'clf__class_weight': ['balanced']    
    },
] 
