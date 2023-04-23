
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend
from keras.layers import Conv2D, Activation, Dense, Dropout, Flatten
from keras.layers import MaxPooling2D
from keras.metrics import accuracy
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from keras.callbacks import Callback
from sklearn.metrics import f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
# Set the seed for reproducibility
from trainmodel import precision

np.random.seed (42)

# Define constants
IMAGE_DIRECTORY = 'dataset/'
INPUT_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 50

# Load the data
no_tumor_images = os.listdir (IMAGE_DIRECTORY + 'no/')
yes_tumor_images = os.listdir (IMAGE_DIRECTORY + 'yes/')
dataset = []
label = []

for i, image_name in enumerate (no_tumor_images):
    if image_name.split ('.')[1] == 'jpg':
        image = cv2.imread (IMAGE_DIRECTORY + 'no/' + image_name)
        image = Image.fromarray (image, 'RGB')
        image = image.resize ((INPUT_SIZE, INPUT_SIZE))
        dataset.append (np.array (image))
        label.append (0)

for i, image_name in enumerate (yes_tumor_images):
    if image_name.split ('.')[1] == 'jpg':
        image = cv2.imread (IMAGE_DIRECTORY + 'yes/' + image_name)
        image = Image.fromarray (image, 'RGB')
        image = image.resize ((INPUT_SIZE, INPUT_SIZE))
        dataset.append (np.array (image))
        label.append (1)

# Convert to numpy arrays
dataset = np.array (dataset)
label = np.array (label)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split (dataset, label, test_size=0.2, random_state=42)

# Normalize the data
x_train = x_train / 255.
x_test = x_test / 255.

# One-hot encode the labels
y_train = to_categorical (y_train, num_classes=2)
y_test = to_categorical (y_test, num_classes=2)

# Data augmentation
datagen_train = ImageDataGenerator (rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

datagen_train.fit (x_train)

# Build the model
model = Sequential ()
model.add (Conv2D (32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3), activation='relu'))
model.add (MaxPooling2D (pool_size=(2, 2)))
model.add (Conv2D (64, (3, 3), activation='relu'))
model.add (MaxPooling2D (pool_size=(2, 2)))
model.add (Conv2D (128, (3, 3), activation='relu'))
model.add (MaxPooling2D (pool_size=(2, 2)))
model.add (Flatten ())
model.add (Dense (128, activation='relu'))
model.add (Dropout (0.5))
model.add (Dense (2, activation='softmax'))

# Compile the model
model.compile (loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision ()])

# Train the model
history = model.fit (datagen_train.flow (x_train, y_train, batch_size=BATCH_SIZE),
                     steps_per_epoch=len (x_train) / BATCH_SIZE,
                     epochs=EPOCHS,
                     validation_data=(x_test, y_test),
                     verbose=1)


class Metrics (Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def precision(y_true, y_pred):
        true_positives = K.sum (K.round (K.clip (y_true * y_pred, 0, 1)))
        predicted_positives = K.sum (K.round (K.clip (y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon ())
        return precision

    def recall(y_true, y_pred):
        true_positives = K.sum (K.round (K.clip (y_true * y_pred, 0, 1)))
        possible_positives = K.sum (K.round (K.clip (y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon ())
        return recall

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray (self.model.predict (self.validation_data[0]))).round ()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score (val_targ.argmax (axis=1), val_predict.argmax (axis=1), average='weighted')
        _val_recall = recall_score (val_targ.argmax (axis=1), val_predict.argmax (axis=1), average='weighted')
        _val_precision = precision_score (val_targ.argmax (axis=1), val_predict.argmax (axis=1), average='weighted')
        self.val_f1s.append (_val_f1)
        self.val_recalls.append (_val_recall)
        self.val_precisions.append (_val_precision)
        logs['val_f1_score'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision


        metrics = Metrics ()
        model.compile (loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',precision, 'f1_score'])
        history = model.fit (x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test),
                     shuffle=False, callbacks=[metrics])

# plot accuracy
plt.plot (history.history['accuracy'], label='accuracy')
plt.plot (history.history['val_accuracy'], label='val_accuracy')
plt.title ('Model Accuracy')
plt.xlabel ('Epoch')
plt.ylabel ('Accuracy')
plt.legend (loc='lower right')
plt.show ()

# plot datapoints
plt.bar (['Training Data', 'Testing Data'], [len (x_train), len (x_test)])
plt.title ('Number of Data Points')
plt.ylabel ('Count')
plt.show ()

# plot precision
plt.plot (history.history['precision'], label='precision')
plt.plot (history.history['val_precision'], label='val_precision')
plt.title ('Model Precision')
plt.xlabel ('Epoch')
plt.ylabel ('Precision')
plt.legend (loc='lower right')
plt.show ()


# Save the model
model.save ('braintumor202350.h')

model.save ('braintumor202350.h5')
