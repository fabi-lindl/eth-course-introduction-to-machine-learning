""" Helper function to load and modify data. """

import tensorflow as tf
from sklearn.model_selection import train_test_split

def create_train_labels(path):
    """
    Read in the training labels, i.e. the provided triplets.
    Create a train test split from these read in triplets and store
    the train and validation triplets into two separate files.
    -----
    path: String to acces the provided data dir.  
    """
    train_data_file = path + 'train_triplets.txt'
    # Read in the samples file. 
    with open(train_data_file, 'r') as file:
        triplets = [sample for sample in file.readlines()]
    
    # Create the train/test split. 
    training_data, validation_data = train_test_split(triplets, test_size=0.2)
    
    # Write train data to file. 
    with open('training_data.txt', 'w') as file:
        for sample in training_data:
            file.write(sample)
    # Write validation data to file. 
    with open('validation_data.txt', 'w') as file:
        for sample in validation_data:
            file.write(sample)
    
    return len(training_data)


def get_image_from_dir(image, train):
    """
    Get image from directory, convert it to a tensorflow datatype and
    implement normalization and data augmentation for training. 
    -----
    image: Loaded image from folder.
    train: Bool 
    """
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    # Normalize. 
    image = image / 127.5 - 1
    # Adjust image resolution to required NN input dimensions. 
    image = tf.image.resize(image, (224, 224))
    if train:
        # Data augmentation for training. 
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_flip_left_right(image)
    return image


def get_image_tuples(path, image_tuple, train):
    """
    Load image 3 tuple from directory and store them as tensorflow datatype.
    Returns the image 3 tuple in tf format.  
    -----
    path: String to provided data directory. 
    image_tuple: Image names/indexes
    train: Bool for train or test settings.
    """
    images = [] # Storage for images. 
    indexes = tf.strings.split(image_tuple)
    # Store images to list. 
    for i in range(3):
        image = get_image_from_dir(tf.io.read_file(path + 'food_images/' + indexes[i] + '.jpg'), train)
        images.append(image)
    if train:
        return tf.stack(images, axis=0), 1
    else:
        return tf.stack(images, axis=0)


def dataset_set_up(fname, path, train=True):
    """
    Sets up a data set from the data stored in the file of fname. 
    The files used are .txt files. 
    """
    ds = tf.data.TextLineDataset(fname)
    ds = ds.map(lambda image_tuple: get_image_tuples(path, image_tuple, train), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds

