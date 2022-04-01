import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics import classification_report

model = tf.keras.models.load_model('model_mnist_logit')

X_test = pickle.load(open('test_images.pkl', 'rb'))
X_test = X_test.reshape(-1, 784)
X_test = X_test/255
y_test = pickle.load(open('test_labels.pkl', 'rb'))

# Load the model and make prediction on testing data
y_pred = model.predict(X_test, batch_size=32, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_bool, output_dict=True))

# eval_loss, eval_acc = model.evaluate(X, y)
#
# print('Evaluation Loss: {:.4f}, Evaluation Accuracy: {:.2f}'.format(eval_loss, eval_acc * 100))
