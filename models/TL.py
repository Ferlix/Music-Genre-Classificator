from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# featurewise_std_normalization
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory('../data/spectrograms512', class_mode="categorical",batch_size=1000, target_size=(512, 512))

labels = (image_data.class_indices)
labels

for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break



X_train, X_test, y_train, y_test = train_test_split(image_batch,
                                                    label_batch,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify = label_batch)

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}

feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3),
                                        name = "pretrained_mobilenet_v2")


feature_extractor_layer.trainable = False

input = layers.InputLayer(input_shape=(512, 512, 3), name = "input_layer")

model = tf.keras.Sequential([input,
  layers.Conv2D(filters = 3, kernel_size = 65, strides = 2),
  feature_extractor_layer,
  layers.Dense(image_data.num_classes, activation='softmax', name = "class_layer")
])

model.summary()

model.compile(
      optimizer=tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999),
      #optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.1, nesterov=True),
      #optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.01),
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

history = model.fit(X_train, y_train, epochs=1000,
                              batch_size = 16,
                              shuffle=True,
                              callbacks = [batch_stats_callback],
                              validation_data=(X_test, y_test))

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1.1])
plt.plot(batch_stats_callback.batch_acc, label= 'Train accuracy', alpha=0.5)

plt.plot(batch_stats_callback.batch_acc_test, label= 'Test accuracy', alpha=0.5)
plt.legend(loc="upper left")
plt.title("Accuracy with augmented data (4800 images)")
plt.grid(True)