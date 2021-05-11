"""
Trains CNN models given a dataset.

A module to train a model for a given dataset, using the desired wavelengths
between one and three layers, an integer number of epochs, and a specified
checkpoint path, where a checkpoint directory will be created if none is
specified, and to evaluate the performance of a model on a given dataset,
outputs an accuracy and loss metric.
"""

# Import Packages
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


def model_learn(dataset_file, wavelength, model_struct, epochs, checkpoint_path,
             load_checkpoint, initial_epoch):
    """
    Train a model given a dataset, returning an accuracy and loss metric.
    
    This function creates a model with the desired structure, the options
    being 1, 2, or 3 layers and 1 or 3 wavelengths. This model is then
    either loaded from an existing checkpoint or trained from scratch.
    The model trains for the desired number of epochs on the input
    dataset, and then saves the updated weights for each training epoch
    in a checkpoint file. In training the model, accuracy and loss metrics
    are output. Accuracy is a measure of how many images in the training
    set the model is correctly identifying and loss is a measure of the
    difference between the desired output for the model and the actual
    output of the model.
    
    Inputs
    ------
        dataset_file    : path to the desired dataset with a label file
                            preferably running from the dataset directory.
        wavelength      : wavelength to use, 'all' for 131, 171, 211 waves
        model_struct    : the model structure to use, '1layer', 'ctmodel'
        epochs          : the number of epochs to train for NOTE: this will
                            train for this amount of ADDITIONAL epochs
        checkpoint_path : path to checkpoint directory, if one does not exist
                            a path will be created,
                            this will also be the name of the model
        load_checkpoint : boolean whether to use the continue from the most
                            recent checkpoint
        initial_epoch   : the initial epoch to start training from if loading
                            from checkpoints

    Outputs
    -------
        None.
    """

    # Load dataset
    train_array = np.load(dataset_file)
    train_labels = np.load(dataset_file.replace('.npy', '_labels.npy'))
    print('Dataset loaded successfully')

    # Model parameters
    image_size = 512
    if wavelength == 'all':
        input_shape = (image_size, image_size, 3)
    else:
        input_shape = (image_size, image_size, 1)
    # avg_pool = True
    # number_layers = 0
    initial_channels = 128
    # conv_channel_increase = 2
    if model_struct == 'ctmodel':
        max_pool_size = (5, 5)
    else:
        max_pool_size = (10, 10)
    kernel_size = (3, 3)
    layer_models = {'1layer': 1, '2layer': 2, 'ctmodel': 3}

    model = models.Sequential()
    for i in range(layer_models[model_struct]):
        kwargs = {}
        if i == 0:
            kwargs['input_shape'] = input_shape
            kwargs['data_format'] = 'channels_last'

        model.add(layers.Conv2D(initial_channels*(i+1), kernel_size,
                                activation='relu',
                                **kwargs))
        model.add(layers.MaxPooling2D(pool_size=max_pool_size))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    # Print the structure of the model
    model.summary()
    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    val_split = 0.2
    batch_size_test = 1

    model_name = checkpoint_path.split('/')[-2]
    print()
    print('model name is:', model_name)
    print()
    checkpoint_filepath = checkpoint_path + model_name + '_{epoch:02d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_filepath)
    # load the weights from the most recent checkpoint if true
    if load_checkpoint == 'True':
        # find the path to the latest checkpoint and load the weights
        latest_checkpt = tf.train.latest_checkpoint(checkpoint_dir)
        print('Latest checkpoint:', latest_checkpt)
        model.load_weights(latest_checkpt)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=True)
    # This means model weights are saved at the end of every epoch

    # Run the model
    history = model.fit(x=train_array, y=train_labels,
                        batch_size=batch_size_test,
                        epochs=epochs + initial_epoch,
                        validation_split=val_split,
                        callbacks=[model_checkpoint_callback],
                        initial_epoch=initial_epoch)

    # Save the history data for later use
    # hist_save_name = './history_dicts/' + model_name + '.npy'

    # Plot the accuracy and loss metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.suptitle(model_name)
    plt.show()

    # print(epochs_range, val_acc)
    # print(epochs_range, acc)
    # print(epochs_range, val_loss)
    return


def model_evaluate(dataset_file, wavelength, model_struct, checkpoint_path,
            start_epoch, end_epoch):
    """
    Evaluate a model given a dataset, returning an accuracy and loss metric.

    This function evaluates the performance of a given model structure,
    with weights loaded from a specified checkpoint. The function either
    outputs the accuracy and loss metric for a single saved checkpoint, or
    outputs a plot showing the accuracy and loss for a range of saved
    checkpoints.

    Inputs
    ------
        dataset_file    : path to the desired dataset with a label file.
                            preferably running from the dataset directory
        wavelength      : wavelength to use, 'all' for 131, 171, 211 waves
        model_struct    : the model structure to use, '1layer', 'ctmodel'
        checkpoint_path : path to checkpoint directory
        start_epoch     : Measure the performance of the model for one or
                            multiple epochs. -1 is the latest epoch and
                            inputting the same value will simply return the
                            accuracy and loss for the chosen model, dataset,
                            and checkpoint. If a range of epochs is input, a
                            graph of the model performance vs epoch will be
                            shown.
                            Note: using 0 0 will evaluate the untrained model
                            structure.
        end_epoch       : the final epoch to load weights from

    Outputs
    -------
        None.

    """

    if start_epoch == end_epoch:
        single_epoch = True
    else:
        single_epoch = False

    # Load dataset
    train_array = np.load(dataset_file)
    train_labels = np.load(dataset_file.replace('.npy', '_labels.npy'))
    print('Dataset loaded successfully')

    # Model parameters
    image_size = 512
    if wavelength == 'all':
        input_shape = (image_size, image_size, 3)
    else:
        input_shape = (image_size, image_size, 1)
    # avg_pool = True
    # number_layers = 0
    initial_channels = 128
    # conv_channel_increase = 2
    if model_struct == 'ctmodel':
        max_pool_size = (5, 5)
    else:
        max_pool_size = (10, 10)
    kernel_size = (3, 3)
    layer_models = {'1layer': 1, '2layer': 2, 'ctmodel': 3}

    model = models.Sequential()
    for i in range(layer_models[model_struct]):
        kwargs = {}
        if i == 0:
            kwargs['input_shape'] = input_shape
            kwargs['data_format'] = 'channels_last'

        model.add(layers.Conv2D(initial_channels*(i+1), kernel_size,
                                activation='relu',
                                **kwargs))
    model.add(layers.MaxPooling2D(pool_size=max_pool_size))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    # Print the structure of the model
    model.summary()
    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    batch_size_test = 1

    model_name = checkpoint_path.split('/')[-2]
    print()
    print('model name is:', model_name)
    print()
    checkpoint_filepath = checkpoint_path + model_name + '_{epoch:02d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_filepath)

    if single_epoch:
        if start_epoch == -1:
            latest_checkpt = tf.train.latest_checkpoint(checkpoint_dir)
            model.load_weights(latest_checkpt)
        else:
            checkpoint_file = checkpoint_path + model_name + '_' + \
                "{0:0=2d}".format(start_epoch) + '.ckpt'
            model.load_weigths(checkpoint_file)

        loss, acc = model.evaluate(x=train_array, y=train_labels,
                                   batch_size=batch_size_test,
                                   verbose=2)
        print('model loss:', loss)
        print('model accuracy:', acc)

    else:
        # does not account for -1 epoch in a range
        epoch_range = range(start_epoch, end_epoch+1)
        loss = np.zeros(len(epoch_range))
        acc = np.zeros(len(epoch_range))
        for i, chkpt_epoch in enumerate(epoch_range):
            checkpoint_file = checkpoint_path + model_name + '_' + \
                "{0:0=2d}".format(chkpt_epoch) + '.ckpt'
            model.load_weights(checkpoint_file)

            loss[i], acc[i] = model.evaluate(x=train_array, y=train_labels,
                                             batch_size=batch_size_test,
                                             verbose=2)
        # Plot the accuracy and loss metrics
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epoch_range, acc, label='Accuracy')
        plt.legend(loc='lower right')
        plt.title('Model Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epoch_range, loss, label='Loss')
        plt.legend(loc='upper right')
        plt.title('Model Loss')
        plt.suptitle(model_name)
        plt.show()

        print('For epochs in the range', epoch_range, 'accuracy is:', acc)
        print('For epochs in the range', epoch_range, 'loss is:', loss)
    return
