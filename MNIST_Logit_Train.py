import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import pickle


NAME = f'MNIST-LOGIT-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

X_train = pickle.load(open('train_images.pkl', 'rb'))
X_train = X_train.reshape(-1, 784)
X_train = X_train/255
y_train = pickle.load(open('train_labels.pkl', 'rb'))
# print(y_train)


model = tf.keras.Sequential([
    # tf.keras.layers.Dense(512, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    # tf.keras.layers.Dense(256, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(256, activation='relu'),
    # tf.keras.layers.Dense(256, activation='relu'),
    # tf.keras.layers.Dense(256, activation='relu'),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

model.build((None, 784))
print(model.summary())

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32, callbacks=[tensorboard])

model.save('model_mnist_logit')

