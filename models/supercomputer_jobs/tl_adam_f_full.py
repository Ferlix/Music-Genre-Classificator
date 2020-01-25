from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib as plt

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().system('pip install -q tf-nightly')
except Exception:
  pass

import tensorflow as tf
import tensorflow_hub as hub
import csv
from tensorflow.keras import layers

## define hyperparameter

optimizer = "adam"          # options:
                            #         adam
                            #         nadam
                            #         SGD
                            #         adagrad

frozen = "frozen"           # options:
                            #         frozen
                            #         unfrozen
        
data_choice = "full_aug"


# Load Training data
print("Load dataset")
# featurewise_std_normalization
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data_train = image_generator.flow_from_directory('/data/s2936860/spectrograms512_train', class_mode="categorical", batch_size=32, target_size=(512, 512))


labels_train = (image_data_train.class_indices)
labels_train


#Load test data
# featurewise_std_normalization
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data_test = image_generator.flow_from_directory('/data/s2936860/spectrograms512_test', class_mode="categorical", batch_size=32, target_size=(512, 512))


labels_test = (image_data_test.class_indices)
labels_test


# The resulting object is an iterator that returns `image_batch, label_batch` pairs.
print("Download the headless model")
# # 3. Download the headless model

#feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4" #@param {type:"string"}
feature_extractor_url  ="/home/s2936860/trained/"


try:
  feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))
except:
  print("An exception occurred")


print('Using mobilenet_v2 v4 as')
print(frozen)

if frozen == "frozen": 
    feature_extractor_layer.trainable = False
else:
    feature_extractor_layer.trainable = True
    

input = layers.InputLayer(input_shape=(512, 512, 3), name = "input_layer")


# # 4. Attach a classification head

print("Create model")
model = tf.keras.Sequential([input,
  layers.Conv2D(filters = 3, kernel_size = 65, strides = 2, name = "extra_input_convolution"),
feature_extractor_layer,
  layers.Dense(image_data_train.num_classes, activation='softmax', name = "class_layer")
])

model.summary()

print("Train model")


# 5. Train the model

print('Using following optimizer:')
print(optimizer)
if optimizer == "adam":
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                                                     loss='categorical_crossentropy',
                                                     metrics=['acc'])
elif optimizer == "nadam":
      model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999),
                                                        loss='categorical_crossentropy',
                                                        metrics=['acc'])

elif optimizer == "SGD":
      model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.1, nesterov=True),
                                                      loss='categorical_crossentropy',
                                                      metrics=['acc'])
elif optimizer == "adagrad":
      model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
                                                      loss='categorical_crossentropy',
                                                      metrics=['acc'])
else:
    optimizer = "SGD"
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.1, nesterov=True),
                                                      loss='categorical_crossentropy',
                                                      metrics=['acc'])



class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []
    self.batch_acc_test = []
    self.batch_losses_test = []


    
  def on_epoch_end (self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.batch_acc_test.append(logs['val_acc'])
    self.batch_losses_test.append(logs['val_loss'])
    self.model.reset_metrics()


batch_stats_callback = CollectBatchStats()

history = model.fit_generator(image_data_train,
                   steps_per_epoch = 25,
                   validation_data = image_data_test,
                   validation_steps = 18,
                   callbacks = [batch_stats_callback],
                   epochs = 100
                   ) 

path = "/home/s2936860/results/"
print('Exporting results')

# Record training progress
with open(path+'training_progress_100epochs_' + optimizer + '_' + frozen + '_' + data_choice+  '.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])

    for line in range(len(batch_stats_callback.batch_losses)): 
        writer.writerow([line,
                         batch_stats_callback.batch_losses[line],
                         batch_stats_callback.batch_acc[line],
                         batch_stats_callback.batch_losses_test[line],
                         batch_stats_callback.batch_acc_test[line],
                         ])
    file.close()

print('Done')

