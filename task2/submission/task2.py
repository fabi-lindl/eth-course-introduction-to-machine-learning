import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.preprocessing import StandardScaler

from data_preparator import *
from feature_modifier import feature_tuning

##############################################################################
# ENABLE THIS BLOCK TO CREATE THE REQUIRED TUNED FEATURES FILES

# Create data with modified features (takes ~ 10 min)
# Depending on the pandas version you use warnings may be 
# printed to the terminal. This is no problem - just ignore them. 
print('Creating tuned train features file ...')
feature_tuning('train')
print('Creating tuned test features file ...')
feature_tuning('test')

##############################################################################

print('Loading data ...')
sc = StandardScaler()
data_used = 'squash'
train_patient_ids, test_patient_ids = [], []
X_train, X_test = [], []
load_patient_data(X_train, train_patient_ids, 'train', data_used)
load_patient_data(X_test, test_patient_ids, 'test', data_used)
# Standardize data. 
X_train = sc.fit_transform(np.array(X_train))
X_test = sc.transform(np.array(X_test))
y_train = []
load_labels(y_train, './data/provided/train_labels.csv')
y_train = np.array(y_train)

# Select the tasks to predict.
print('Training model ...')
tasks_dict = {
    'med_tests': [1, 11],
    'sepsis': [11, 12],
    'vitals': [12, 16],
    'overall': [1 ,16]
}
tasks_list = ['med_tests', 'sepsis', 'vitals']  

# Predict values for each label. 
num_samples = len(X_test)
predictions = np.zeros((1, num_samples))
for t in tasks_list:
    print(f'Task: {t}')
    start = tasks_dict[t][0]
    end = tasks_dict[t][1]
    for label_index in range(start, end):
        # Extract the current label. 
        y_train_selected = y_train[:, label_index]
        # Fit the data. 
        if t == 'vitals':
            clf = HistGradientBoostingRegressor().fit(X_train, y_train_selected)
            y_pred = clf.predict(X_test)
            predictions = np.vstack((predictions, y_pred))
        else:
            clf = HistGradientBoostingClassifier().fit(X_train, y_train_selected)
            y_pred = clf.predict_proba(X_test)
            predictions = np.vstack((predictions, y_pred[:, 1]))
# Delete the first row vector, i.e. the artifical zeros vector.
predictions = np.delete(predictions, 0, 0)

# Create csv file with the predicted labels. 
print('Writing predictions to csv file ...')
predictions = np.transpose(predictions)
predictions = np.round(predictions, decimals=3)
preds = [] # Predicted labels and ground truth storage.
# Create header labels according to the target prediction labels.
pid = ['pid']
tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST',
         'LABEL_Alkalinephos', 'LABEL_Bilirubin_total', 'LABEL_Lactate',
         'LABEL_TroponinI', 'LABEL_SaO2', 'LABEL_Bilirubin_direct',
         'LABEL_EtCO2']
sepsis = ['LABEL_Sepsis']
vitals = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
header_labels = pid + tests + sepsis + vitals

# Store results to file
path = './'
fname_preds = 'prediction.csv' 
with open(path + fname_preds , 'w', newline='') as f:
    wr = csv.writer(f, delimiter=',')
    wr.writerow(header_labels)
    cnt = 0
    for pred in predictions:
        writeout = []
        writeout.append(test_patient_ids[cnt]) # Patient id
        writeout += list(pred) # Predicted labels
        wr.writerow(writeout)
        cnt+=1
