import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('model_mnist_logit')


img_path_test = input("Please enter the path of the image: ")
img_arr_test = cv2.imread(img_path_test)
img_arr_test = cv2.cvtColor(img_arr_test, cv2.COLOR_BGR2GRAY)
img_arr_test = img_arr_test.reshape(-1, 784)
img_arr_test = img_arr_test/255

prediction = model.predict(img_arr_test)
print(prediction)
prediction = np.argmax(prediction)
print(prediction)
