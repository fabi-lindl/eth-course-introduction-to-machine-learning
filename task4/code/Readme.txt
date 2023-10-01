Dear TA,

Please follow the below instructions so that you can
get the code to work quickly and easily. 

 - Copy all the food images into the food_images folder
   Images must be unzipped for the code to work correctly. 

 - The test_triplets.txt file resides in ./data/provided/
   For a test run of both the public test triplets and the
   private triplets put the .txt file with this exact name
   "test_triplets.txt" file into this folder. 
   The code will load it from this directory. 

 - Inside the config_tf.py file there is a parameter named
   'predict_only'. If this parameter is set to 'True' the code
   will load the stored checkpoint and assign these parameters
   to the model. No model training is going to be carried
   out in this case. 
   
 - Unfortunately, I cannot provide you with the checkpoint as
   the server has a 1 MB upload limit and the checkpiont has
   a size of about 30 MB. 
   You can first run the code to predict the public labels again,
   which creates a checkpoint, too. 
   For the second run on the private test set you can set
   'predict_only = True' in config_tf.py to make use of the stored
   checkpoint for this second run. 
   A message is going to be printed if the checkpoint is 
   loaded successfully. 

 - The predicted .csv files are stored in ./data/results/


Best regards

