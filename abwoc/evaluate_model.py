# A script to evaluate the performance of a model on a given dataset, outputs
# an accuracy and loss metric

# Example Call: $evaluate_model.py
#       './validation_sets/true_validation_171_512_normalized.npy'
#       '171' '1layer' './final_checkpoints/onewave_131_512_1800_1layer/' -1 -1
#   This would load the desired dataset, and evaluate the chosen models
#   performance using the chosen checkpoint weights

# Imports
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Arguments given to script
dataset_file = sys.argv[1]  # path to the desired dataset with a label file
#                           preferably running from the dataset directory
wavelength = sys.argv[2]  # wavelength to use, 'all' for 131, 171, 211 waves
model_struct = sys.argv[3]  # the model structure to use, '1layer', 'ctmodel'
checkpoint_path = sys.argv[4]  # path to checkpoint directory
#   Measure the performance of the model for one or multiple epochs
#   -1 is the latest epoch and inputting the same value will simply return the
#   accuracy and loss for the chosen model, dataset, and checkpoint. If a range
#   of epochs is input, a graph of the model performance vs epoch will be shown
#   Note: using 0 0 will evaluate the untrained model structure
start_epoch = int(sys.argv[5])  # the training epoch to load weigths from
end_epoch = int(sys.argv[6])  # the final epoch to load weights from

# Print and check inputs

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
avg_pool = True
number_layers = 0
initial_channels = 128
conv_channel_increase = 2
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
    # cannot currently account for -1 epoch in a range
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
