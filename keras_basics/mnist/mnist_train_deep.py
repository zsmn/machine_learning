import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# loading database
mnist = tf.keras.datasets.mnist

# loading train and test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizing
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_train, axis = 1)

# num pixels
num_pixels = x_train.shape[1] * x_train.shape[2]

# create model
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())

    # hidden layer 1 (128)
    model.add(tf.keras.layers.Dense(128, kernel_initializer = 'normal', activation='relu'))
    # hidden layer 2 (256)
    model.add(tf.keras.layers.Dense(256, kernel_initializer = 'normal', activation='relu'))

    # output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

model = create_model()
model.fit(x_train, y_train, epochs=3)

# save model
model.save('save_model.model')

# test here with some data
plt.imshow(x_test[1], cmap = plt.cm.binary)
plt.show()

predictions = model.predict([x_test])
print(np.argmax(predictions[1]))
