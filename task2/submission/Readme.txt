
The following steps are necessary to run the code successfully. 

First:
---------------------------------------
Store the provided files
 
   - test_features.csv
   - train_features.csv
   - train_labels.csv

in the folder ./data/provided/


Second: 
---------------------------------------
Run the file 

   - task2_hgb_submit.py

The file prints statements about the current progress of the code. 
Firstly, the provided datafiles are read in and used to create new
csv files with adjusted features. 
This feature creation process takes ~ 5-10 min depending on your 
computer. 
Depending on your pandas version an error message might be printed in 
your terminal. Do not worry about this, the code works correct, the
warning is generated due to different pandas versions in use. 


Third:
---------------------------------------
The predicted csv file is stored in the same folder in which 

   - task2_hgb_submit.py

resides. 


Sidenote:
---------------------------------------
The file

    - feature_information_2.csv

stores estimated values for patient features. 
Do not touch this file. 
