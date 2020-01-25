from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
import csv


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print()

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


# Load Training data
print("Load dataset")
# featurewise_std_normalization
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data_train = image_generator.flow_from_directory('../data/dataset/spectrograms512_train', class_mode="categorical", batch_size=16, target_size=(512, 512))


labels_train = (image_data_train.class_indices)
labels_train


#Load test data
# featurewise_std_normalization
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data_test = image_generator.flow_from_directory('../data/dataset/spectrograms512_test', class_mode="categorical", batch_size=16, target_size=(512, 512))


labels_test = (image_data_test.class_indices)
labels_test

print("Load and create model")

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

feature_extractor_layer = hub.KerasLayer(feature_extractor_url)


feature_extractor_layer.trainable = True


model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dropout(rate=0.2),
  layers.Dense(image_data_test.num_classes, activation='softmax', name = "class_layer")
])

model.build((None,512,512,3))
model.summary()

model.compile(
      optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
      loss='categorical_crossentropy',
      metrics=['acc'])


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []
        self.batch_acc_test = []
        self.batch_losses_test = []

    def on_epoch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.batch_acc_test.append(logs['val_acc'])
        self.batch_losses_test.append(logs['val_loss'])
        self.model.reset_metrics()


batch_stats_callback = CollectBatchStats()

print("Start training process")
history = model.fit_generator(image_data_train,
                   steps_per_epoch = 25,
                   validation_data = image_data_test,
                   validation_steps = 18,
                   callbacks = [batch_stats_callback],
                   epochs = 100
                   )

path = '../data/'
print('Exporting results')

# Record training progress
with open(path+'results.csv', 'w', newline='') as file:
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