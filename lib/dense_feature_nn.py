 #!/usr/bin/env python -W ignore::DeprecationWarning
from data_preprocess import FiducialDataProcess
import ipdb
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

my_data = np.genfromtxt('processed_featuredata_nn_int.csv', delimiter=',')
np.random.seed(0)
np.random.shuffle(my_data)

features_init = my_data[:,1:]
features = features_init / features_init.max(axis=0)
labels = my_data[:,0] 
labels -= np.ones(len(labels))

train_labels, train_data = labels[0:2000], features[0:2000,:]
test_labels, test_data = labels[2000:], features[2000:,:]

#tf_data = tf.convert_to_tensor(my_data)
#tf.random.shuffle(tf_data, seed = 1)
from tensorflow.keras import optimizers

adam = optimizers.Adam(lr=0.05)

model = keras.Sequential([
	keras.layers.Dense(1024, activation = tf.nn.relu),
	#keras.layers.Dropout(0.5),
	keras.layers.Dense(1024, activation = tf.nn.relu),
	#keras.layers.Dropout(0.5),
	keras.layers.Dense(256, activation = tf.nn.relu),
	keras.layers.Dense(128, activation = tf.nn.relu),
	keras.layers.Dense(64, activation = tf.nn.relu),
	#keras.layers.Dropout(0.2),
    keras.layers.Dense(22, activation = tf.nn.softmax)
])

model.compile(optimizer = adam,
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

history = model.fit(train_data, train_labels, validation_data = (test_data, test_labels), batch_size = 64,epochs = 100)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get the number of epochs
epochs = range(len(acc))

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, color='blue', label='Train')
plt.plot(epochs, val_acc, color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

_ = plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, color='blue', label='Train')
plt.plot(epochs, val_loss, color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()