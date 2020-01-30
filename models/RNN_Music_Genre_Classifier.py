data_storage_path = 'ML_experiments'
path_train_data_set = 'dataset_transformed/spectrograms512_train'
path_test_data_set = 'dataset_transformed/spectrograms512_test'


"""# Imports

Specific tensorflow setup to make sure it's gonna run on GPU
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
	print('GPU device not found. Device: ' + str(device_name))
else:
	print('Found GPU at: {}'.format(device_name))

# Import os
import os

from shutil import copyfile  # Making copy of this file instance (including param settings used)

# Imports tensorflow
from tensorflow.keras import datasets, layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import pathlib  # For data generators 

# Imports image handling
import numpy as np

# Save training progress
import csv
from datetime import datetime

# Set up folder for data gathering during training process
now = datetime.now()
TIME_STAMP = now.strftime("_%Y_%d_%m__%H_%M_%S__%f")
MODEL_ID = 'Model_' + TIME_STAMP
training_path = data_storage_path + '/Trained_Models/RNN_Models/'
path = training_path + MODEL_ID + '/'

if not os.path.exists(path):
    os.makedirs(path)
    print('Created dir: ' + path)
else:
    path = None
    raise Exception('PATH EXISTS!')


# Save a copy of this file - Including current param settings - For convenience during param sweeping
src = os.path.realpath(__file__)
dst = path + 'model_copy.py'
copyfile(src, dst)


"""# Set up the CNN architecture & helpers

## Define Model Architecture

### Model
"""

# Parameters
dimensions = 512  # Input dimension: 512x512
units = 512       # Dimensionality of RNN output tensor
classes = 10      # Number of output nodes in final layers (=nr of distinct classes)

# Squeezing layer to transform (32, 512, 512, 1) to (32, 512, 512)

class Squeezer(layers.Layer):
  def __init__(self):
    super(Squeezer, self).__init__()

  def build(self, input_shapes):
    pass
  
  def call(self, input):
    return tf.squeeze(input, axis=3)

# Reset tf sessions
tf.keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.


############### Define RNN layers ###############

# Full, efficient RNN implementation

full_rnn_cell_1 = layers.GRU(units, # How many output units (per iteration of RNN)    # Formerly: SimpleRNN
                                   input_shape=(1, dimensions), # 1 row of 512 pixels
                                   kernel_regularizer=regularizers.l2(0.0001),
                                   recurrent_regularizer=regularizers.l2(0.0001),
                                   bias_regularizer=regularizers.l2(0.0001), 
                                   return_sequences=True  #return_sequences = X outputs per input image's row, here X = dim(image) = 512
         ) 
full_rnn_cell_2 = layers.GRU(units, # How many output units (per iteration of RNN)    # Formerly: SimpleRNN
                                   input_shape=(1, dimensions), # 1 row of 512 pixels
                                   kernel_regularizer=regularizers.l2(0.0001),
                                   recurrent_regularizer=regularizers.l2(0.0001),
                                   bias_regularizer=regularizers.l2(0.0001), 
                                   return_sequences=True  #return_sequences = X outputs per input image's row, here X = dim(image) = 512
         ) 

full_rnn_cell_3 = layers.GRU(units, 
                                   input_shape=(1, dimensions),
                                   kernel_regularizer=regularizers.l2(0.0001),
                                   recurrent_regularizer=regularizers.l2(0.0001),
                                   bias_regularizer=regularizers.l2(0.0001),
                                   return_sequences=False
         )

#Alternatively: layers.SimpleRNN || layers.LSTM || layers.GRU



############### Define model ###############

# Set up model architecture in terms of its layers
model = models.Sequential()

model.add(Squeezer())

# Add layers
model.add(full_rnn_cell_1)
model.add(full_rnn_cell_2)
model.add(full_rnn_cell_3)

model.add(layers.Dense(516, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
#model.add(layers.Dropout(0.1))
model.add(layers.Dense(classes, activation='softmax'))

# Note on regularizer(s), copied from https://www.tensorflow.org/tutorials/keras/overfit_and_underfit:
# l2(0.001) means that every coefficient in the weight matrix of the layer will add 0.001 * weight_coefficient_value**2
# to the total loss of the network.

# Compile model & make some design choices
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001,
                                           beta_1=0.9,
                                           beta_2=0.999,
                                           epsilon=1e-07,
                                           amsgrad=False,
                                           name='Adam'
                                           ),
              loss='sparse_categorical_crossentropy',  # Capable of working with regularization
              metrics=['accuracy', 'sparse_categorical_crossentropy'])

# Construct computational graph with proper dimensions
inputs = np.random.random([32, 512, 512, 1]).astype(np.float32)
model(inputs)

# Print summary
model.summary()

"""## Define Callbacks"""

# Definition of callbacks adjusted from https://www.tensorflow.org/guide/keras/train_and_evaluate

early_stopping_callback = EarlyStopping(
        monitor='val_accuracy',    # Stop training when `val_loss` is no longer improving
        min_delta=0,               # "no longer improving" being defined as "no better than 0|5e-1 less"
        patience=2,                # "no longer improving" being further defined as "for at least 2 epochs"
        verbose=0)                 # Quantity of printed output

model_saving_callback = ModelCheckpoint(
        filepath=path+'cnn_model.h5',
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        save_best_only=True,
        monitor='val_accuracy',
        # mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
        # overwrite the current save file is made based on either the maximization
        # or the minimization of the monitored quantity. For `val_acc`, this
        # should be `max`, for `val_loss` this should be `min`, etc. In `auto`
        # mode, the direction is automatically inferred from the name of the
        # monitored quantity.
        verbose=0)

# Join list of required callbacks
callbacks = [model_saving_callback] # Outtake: early_stopping_callback

"""## Define Data Generator

**Pre**-Processing
"""

def preprocessing_function(x):
    """
    Rotating image, such that time evolves over rows: 0'th row := 0'th time step, 
    512'th row := 512'th time step. Frequencies increasing from left image
    border to right one. 
    """
    assert x.shape == (512, 512, 1)
    # Rotate by 270° (=3*90°) 
    return np.rot90(x, 3)

"""### Training data generator"""

# The 1./255 is to convert from uint8 to float32 in range [0,1].
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, 
                                                     preprocessing_function=preprocessing_function  # Pre-processing function may be passed here
                                                     )

BATCH_SIZE = 32
IMG_HEIGHT = 512
IMG_WIDTH = 512
STEPS_PER_EPOCH = 25
data_dir = path_train_data_set
data_dir = pathlib.Path(data_dir)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])


train_data_gen = train_image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     color_mode='grayscale',  # Make sure that BW images are read in (indeed) as BW
                                                     class_mode='sparse', # Class represented by 1 integer (instead of categorical==1-hot-encoding)
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))

"""### Test|Evaluation data generator"""

# The 1./255 is to convert from uint8 to float32 in range [0,1].
test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, 
                                                     preprocessing_function=preprocessing_function
                                                     )

BATCH_SIZE = 32
IMG_HEIGHT = 512
IMG_WIDTH = 512
STEPS_PER_EPOCH = 25 #np.ceil(image_count/BATCH_SIZE)
data_dir = path_test_data_set
data_dir = pathlib.Path(data_dir)


CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])


test_data_gen = test_image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     color_mode='grayscale',
                                                     class_mode='sparse',
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))


"""# Perform training

'accuracy' == accuracy achieved during training on training data

'val_accuracy' == accuracy achieved on Test/Evaluation data
"""

#assert tf.config.list_physical_devices('GPU')
#assert tf.test.is_built_with_cuda()

# Set number of desired epochs
epochs = 100

# Perform x epochs of training
with tf.device(device_name):
	history = model.fit(
		x=train_data_gen,
		#y=None,
		#batch_size=None,
		epochs=epochs,
		verbose=1,
		callbacks=callbacks,
		#validation_split=0.0,
		validation_data=test_data_gen,
		shuffle=True,
		#class_weight=None,
		#sample_weight=None,
		initial_epoch=0,
		steps_per_epoch=25,
		validation_steps=18,
		#validation_freq=1,
		max_queue_size=2,
		#workers=1,
		#use_multiprocessing=False,
		#**kwargs
	)

# Save the entire model as a final model to a HDF5 file.
name = 'final_model'
model.save(path+name+'.h5')

# Record training progress
with open(path+'training_progress.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy", "sparse_categorical_crossentropy"])
    for line in range(len(history.history['loss'])): 
        epoch = str(line+1)
        writer.writerow([epoch,
                         history.history["loss"][line], 
                         history.history["accuracy"][line], 
                         history.history["val_loss"][line], 
                         history.history["val_accuracy"][line], 
                         history.history["sparse_categorical_crossentropy"][line]
                         ])
    # Save some more important bits/summary
    writer.writerow(["End of training. Summary:"])
    writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy", "sparse_categorical_crossentropy"])
    # Max accuracy
    writer.writerow(["Max accuracy row:"])
    x = np.argmax(history.history["accuracy"])
    writer.writerow([str(x+1),
                         history.history["loss"][x], 
                         history.history["accuracy"][x], 
                         history.history["val_loss"][x], 
                         history.history["val_accuracy"][x], 
                         history.history["sparse_categorical_crossentropy"][x]
                         ])
    # Max val_accuracy
    writer.writerow(["Max val_accuracy row:"])
    x = np.argmax(history.history["val_accuracy"])
    writer.writerow([str(x+1),
                         history.history["loss"][x], 
                         history.history["accuracy"][x], 
                         history.history["val_loss"][x], 
                         history.history["val_accuracy"][x], 
                         history.history["sparse_categorical_crossentropy"][x]
                         ])
    file.close()

print('Done.')
