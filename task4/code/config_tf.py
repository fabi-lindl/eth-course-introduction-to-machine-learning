
""" Define frequently used variables. """

# Hyperparameters for the NN. 
epochs = 5
bs = 32
test_bs = 1

# Paths. 
colab = False
if colab == False:
    path_provided = './data/provided/'
    path_results = './data/results/'
    path_checkpoint = './data/checkpoint/'
else:
    path_provided = '/content/drive/MyDrive/ETH/IML/Projects/task4/task4/data/provided/'
    path_results = '/content/drive/MyDrive/ETH/IML/Projects/task4/task4/data/results/'
    path_checkpoint = '/content/drive/MyDrive/ETH/IML/Projects/task4/task4/data/checkpoint/'

# Predict and/or train.
predict_only = False
