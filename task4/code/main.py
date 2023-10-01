""" Neural network training pipeline. """

import tensorflow as tf
import numpy as np

# Import custom files. 
from data_preparation_tf import *
import config_tf


""" Helper functions. """

def image_distances(features):
    """ Calculate the squared distance between the image tuple. """
    out0 = features[..., 0]
    out1 = features[..., 1]
    out2 = features[..., 2]
    d1 = tf.reduce_sum(tf.square(out0 - out1), 1)
    d2 = tf.reduce_sum(tf.square(out0 - out2), 1)
    return d1, d2


def triplet_loss(_, features):
    """ Compute the triplet loss. """
    d1, d2 = image_distances(features)
    delta_d = d1 - d2
    return tf.reduce_mean(tf.math.softplus(delta_d))


def acc(_, features):
    """
    Check if the distance of the dissimilar image is greate or equal to the similar image.
    """
    d1, d2 = image_distances(features)
    return tf.reduce_mean(tf.cast(tf.greater_equal(d2, d1), tf.float32))


""" Neural network models. """

def nn_model(freeze_pretrained_net=False):
    """
    Neural network model, siamese network.
    """
    resolution = 224
    inputs = tf.keras.Input(shape=(3, resolution, resolution, 3))
    # Feature extractor network. 
    encoder = tf.keras.applications.DenseNet121(
        include_top=False,
        input_shape=(resolution, resolution, 3)
    )
    # Decide whether the whol net should be trained. 
    encoder.trainable = freeze_pretrained_net
    # Fully connected layers. 
    decoder = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))
    ])
    # Extract nework inputs. 
    image0, image1, image2 = inputs[:, 0, ...], inputs[:, 1, ...], inputs[:, 2, ...]
    # Network pipeline. 
    out0 = decoder(encoder(image0))
    out1 = decoder(encoder(image1))
    out2 = decoder(encoder(image2))
    outputs = tf.stack([out0, out1, out2], axis=-1)
    net = tf.keras.Model(inputs=inputs, outputs=outputs)
    return net


def prediction_net(net):
    """ Prediction net """
    d1, d2 = image_distances(net.output)
    predictions = tf.cast(tf.greater_equal(d2, d1), tf.int8)
    return tf.keras.Model(inputs=net.inputs, outputs=predictions)



if '__main__' == __name__:
    
    # Create training and validation datasets from the provided samples file. 
    if config_tf.predict_only == False:
        num_train_samples = create_train_labels(config_tf.path_provided)
        train_ds = dataset_set_up('training_data.txt', config_tf.path_provided)
        val_ds = dataset_set_up('validation_data.txt', config_tf.path_provided)

        # Create Datasets. 
        train_ds = train_ds.shuffle(1024, reshuffle_each_iteration=True).repeat().batch(config_tf.bs)
        val_ds = val_ds.batch(config_tf.bs)
    
    test_ds = dataset_set_up(config_tf.path_provided + 'test_triplets.txt', config_tf.path_provided, train=False).batch(config_tf.test_bs).prefetch(2)

    # Design model. 
    net = nn_model()
    optim = tf.keras.optimizers.Adam(learning_rate=0.001)
    net.compile(optimizer=optim, loss=triplet_loss, metrics=[acc])

    if config_tf.predict_only == False:
        # Perform model training. 

        # Create a callback that saves the model's weights. 
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=config_tf.path_checkpoint,
            save_weights_only=True,
            verbose=1
        )

        # Train the model. 
        fit_out = net.fit(
            train_ds,
            steps_per_epoch=int(np.ceil(num_train_samples / config_tf.bs)),
            epochs=config_tf.epochs,
            validation_data=val_ds,
            validation_steps=1,
            callbacks=[cp_callback],
        )
    else:
        # Load stored parameters from our model training. 
        net.load_weights(config_tf.path_checkpoint)
        print('Model parameters loaded successfully.')
    
    # Predict test data. 
    prediction_net = prediction_net(net)
    num_test_tuples = 59544 # Number of test samples, counted from .txt file.
    num_steps = int(np.ceil(num_test_tuples / config_tf.test_bs))
    preds = prediction_net.predict(test_ds, steps=num_steps, verbose=1)
    np.savetxt(config_tf.path_results + 'results.csv', preds, fmt='%i')

