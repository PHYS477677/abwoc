# A script to train a model for a given dataset, using the desired wavelengths
# between one and three layers, an integer number of epochs, and a specified
# checkpoint path, where a checkpoint directory will be created if none is
# specified

# Example Call: $train_model.py
#     "./training_sets/training_set_171_512_normalized.npy" '171' '1layer' 5
#     './final_checkpoints/onewave_171_512_1800_ctmodel/' False 0
#   This would load the training_set_171_512_normalized dataset, and train
#   a 1 layer model for 5 epochs, saving checkpoints in the input directory

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
epochs = int(sys.argv[4])  # the number of epochs to train for
#                   NOTE: this will train for this amount of ADDITIONAL epochs
checkpoint_path = sys.argv[5]  # path to checkpoint directory, if one does not
#                                exist a path will be created, this will also
#                                be the name of the model
load_checkpoint = sys.argv[6]  # boolean whether to use the continue from the
#                               most recent checkpoint
initial_epoch = int(sys.argv[7])  # the initial epoch to start training from if
#                           loading from checkpoints

# Print and check inputs

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
hist_save_name = './history_dicts/' + model_name + '.npy'

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
