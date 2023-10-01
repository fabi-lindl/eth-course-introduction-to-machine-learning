""" Functions to load and prepare the data. """
import csv

""" Load data from .csv files. """

def load_patient_data_csv_to_list(patients, pids, path, add_column=False):
    """
    Read in data from CSV file.
    Store the read in data as dictionary values of each patient.
    The dict value is a list of lists, where each list represents measurments at a
    certain point in time. 
    patients: Empty dictionary. 
    pids: Empty list to store the patient ids to. 
    path: Path to file (string). 
    """ 
    # Start value for indexing (Baris' file contains an additional first column)
    if add_column:
        k = 1
    else:
        k = 0
    with open(path, newline='') as csvfile:
        tdr = csv.reader(csvfile, delimiter=',')
        next(tdr) # Skip the first line (header information). 
        pid = 'x'
        cnt = -1
        for sample in tdr:
            id_ = sample[k]
            if id_ != pid:
                cnt+=1
                pid = id_
                pids.append(int(id_)) # Store patient id.
                patients[cnt] = []
            patients[cnt].append([float(i) if i != 'nan' else None for i in sample[k:]])

def load_patient_data(X, pids, tr_or_val, option):
    """
    Load the patient samples from the csv file. Stores the samples as a list of lists. 
    One list for each patient. The 12 data points are either reduced to one vector or
    all time measurements are concatenated to one large vector. 
    --------
    X: Empty list to store the loaded data to.
    pids: Empty list to store patient ids to.  
    tr_or_val: String, either 'train' or 'val' to decide which data to load.
    option   : String, either 'sqash' or 'all_time' to decide which vector to create. 
    """
    # Load patient data. 
    patients = {}
    if tr_or_val == 'train':
        path = './data/filled/train/'
        load_patient_data_csv_to_list(patients, pids, path+'train_features_filled_submit.csv', add_column=True)
    else:
        path = './data/filled/test/'
        load_patient_data_csv_to_list(patients, pids, path+'test_features_filled_submit.csv', add_column=True)        

    # Transform the data. 
    # Either use only one value per label or use one label for each of the 12 hours. 
    patients_transformed = {}
    if option == 'squash':
        reduce_patients_to_one_row(patients, patients_transformed, 'mean')
        zero_nan_values(patients_transformed, X)
    else:
        time_data_to_single_feature_vector(patients, X)

def load_labels_csv_to_list(patient_labels, path, option):
    """
    Loads training data labels from csv file to a list.
    patient_labels: List to store the csv data to.
    option        : String to specify which labels to load.
                    'med_tests', 'sepsis' or * (wildcard)
    """
    # Choose data to extract from each lable.
    if option == 'med_tests':
        # Probabilities of  patient receiving a particular medicine during the remaining stay. 
        start = 1
        end = start+10
    elif option == 'sepsis':
        # Probability that the patient is likely to have a sepsis. 
        start = 11
        end = start+1
    else:
        # Loads the patient data for life threating events. 
        start = 12
        end = start+4
    # Read file. 
    with open(path, newline='') as csvfile:
        labels = csv.reader(csvfile, delimiter=',')
        next(labels)
        for label in labels:
            l = [float(label[0])]
            l += [float(i) for i in label[start:end]]
            patient_labels.append(l)
    # Sort patient labels by patient id and discard patient id. 
    patient_labels.sort()
    for i in patient_labels:
        del i[0]

def load_labels(patient_labels, path):
    with open(path, newline='') as csvfile:
        labels = csv.reader(csvfile, delimiter=',')
        next(labels)
        for label in labels:
            l = [float(i) for i in label]
            patient_labels.append(l)
    # Sort patient labels by patient id and discard patient id. 
    patient_labels.sort()


""" Print information on the data set. """

def print_loaded_data(num_features, num_measurements, num_patients):
    """ Print the input values to the terminal. """
    print('----------------------------\nData dimensions')
    print(f'Patients: {num_patients}\nMeasurements: {num_measurements}\n' + 
          f'Features: {num_features}')
    print('----------------------------')


""" Deal with nan values. """

def zero_nan_values(patients_time_reduced, rlist):
    """
    Extracts the reduced time feature vectors from a patient dict and stores 
    them into a list of lists, with zeroed NaN values. 
    patients_time_reduced: Patient dictionary with time reduced feature vectors. 
    rlist: List to store the feature vectors to (with NaNs changed to zeros).
    """
    for key, patient in patients_time_reduced.items():
        rlist.append([0.0 if feature == None else feature for feature in patient])
        del rlist[-1][0] # Delete patient id entry. 


""" Transform patient data into vectors that are used for model training. """

def reduce_patients_to_one_row(patients, patients_time_reduced, option='mean'):
    """
    Reduce each patient's twelve our data maeasurements to only one sample row.
    Reduce the 12 values of one feature into one feature. 
    If option = 'mean' the mean of existing values is used. 
    If option = 'median' the median of existing values is used.
    patients: Dictionary, values is a list of lists, the lists contain the patient
              for each time step
    patients_time_reduced: Empty dictionary. 
    """
    num_measurements = len(patients[0])
    num_features = len(patients[0][0])
    feature_limit = num_features-1
    cnt = 0
    num_not_none = 0
    for patient in patients.values():
        # Patient's one line data. 
        patients_time_reduced[cnt] = [None for i in range(feature_limit)]  
        # Patient data (skip the time information at index 1)
        patients_time_reduced[cnt][0] = patient[0][0] # Id.
        patients_time_reduced[cnt][1] = patient[0][2] # Age.  

        for feature in range(3, feature_limit):
            for measurement in range(num_measurements):
                m = patient[measurement][feature]
                if m != None:
                    num_not_none+=1
                    if patients_time_reduced[cnt][feature] != None:
                        patients_time_reduced[cnt][feature] += m
                    else:
                        patients_time_reduced[cnt][feature] = m
            # Compute the mean of this feature. 
            if patients_time_reduced[cnt][feature] != None:
                patients_time_reduced[cnt][feature] /= num_not_none
            num_not_none = 0
        cnt+=1

def time_data_to_single_feature_vector(patients, X):
    """
    Combines the 12 feature vectors of each patient into one giant feature vector for a patient. 
    patients: Dictionary, values is a list of lists, the lists contain the patient
              for each time step.
    X: Empty list, gets filled with lists that store all features of one patient 
       (all time features separately). 
    """
    cnt = 0
    for patient in patients.values():
        # First measurement. 
        X.append(patient[0][2:]) # Skip patient id [0] and measurement time [1]
        # All other measurements. 
        limit = len(patient)
        for i in range(1, limit):
            X[cnt] += patient[i][3:]
        cnt+=1
