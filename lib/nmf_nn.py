from data_preprocess import FiducialDataProcess
import ipdb
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import optimizers

nmf_features  = np.load('nmf_features_3003.npy',allow_pickle=True)
x_train_nmf = nmf_features[0]
y_train = nmf_features[1]#.to_numpy()
y_train -= np.ones(len(y_train), dtype='int64')

x_test_nmf = nmf_features [2]
y_test = nmf_features[3]#.to_numpy()
y_test -= np.ones(len(y_test), dtype='int64')

#adam = optimizers.Adam(lr=0.05)

model = keras.Sequential([
	keras.layers.Dense(96, activation = tf.nn.leaky_relu),
	keras.layers.Dropout(0.4),
	keras.layers.Dense(64, activation = tf.nn.leaky_relu),
	keras.layers.Dropout(0.3),
	keras.layers.Dense(48, activation = tf.nn.leaky_relu),
	keras.layers.Dropout(0.2),
    keras.layers.Dense(22, activation = tf.nn.softmax)
])

model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

history = model.fit(x_train_nmf, y_train, validation_data = (x_test_nmf,y_test), batch_size = 24,epochs = 50)


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
plt.show()

_ = plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, color='blue', label='Train')
plt.plot(epochs, val_loss, color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()