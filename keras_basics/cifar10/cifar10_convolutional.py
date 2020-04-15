import sys
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
import numpy as np
from matplotlib import pyplot
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as k
from keras.optimizers import SGD

k.common.set_image_data_format('channels_first')

# Criação dos subconjuntos de treino e teste
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

image_width = 32
image_height = 32
channels = 3 # rgb

# Aplicando reshape no conjunto de dados
if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], channels, image_width, image_height)
    X_test = X_test.reshape(X_test.shape[0], channels, image_width, image_height)
    input_shape = (channels, image_width, image_height)
else:
    X_train = X_train.reshape(X_train.shape[0], image_width, image_height, channels)
    X_test = X_test.reshape(X_test.shape[0], image_width, image_height, channels)
    input_shape = (image_width, image_height, channels)

# Amenizar o uso de memoria (com float 32)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalizamos os valores de acordo com a variação da escala cinza (0 - 255)
X_train /= 255.0
X_test /= 255.0

# Aplicamos a solucao de one-hot-coding para classificação multiclasses
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# Numero de tipos de digitos encontrados no MNIST
num_classes = Y_test.shape[1]

def deeper_cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Uma camada de DropOut com 20% de probabilidade
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Uma camada de DropOut com 20% de probabilidade
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Uma camada de DropOut com 20% de probabilidade
    model.add(Dropout(0.2))

    # Uma camada de Flatten preparando os dados para a camada fully connected (final da conv network)
    model.add(Flatten())

    # Camada fully connected de 128 neurônios
    model.add(Dense(128, activation = 'relu', kernel_initializer='he_uniform'))
    # Uma camada de DropOut com 20% de probabilidade
    model.add(Dropout(0.2))

    # A camada de saida possui o numero de neuronios compativel com o numero de classes a serem classificadas
    # com uma função de ativação do tipo softmax
    model.add(Dense(num_classes, activation = 'softmax', name = 'preds'))

    # Por fim, compilando o modelo
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

    return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

model = deeper_cnn_model()

# O método summary revela quais são as camadas que formam o modelo, seus formatos e o numero
# de parametros envolvidos em cada etapa
model.summary()

# Processo de treinamento do modelo
model_log = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 10, batch_size = 64)

# Avaliação de performance do modelo
scores = model.evaluate(X_test, Y_test, verbose = 0)
print("Acurácia de: %.2f%%" % (scores[1] * 100))
print("Erro de: %.2f%%" % (100 - scores[1] * 100))
summarize_diagnostics(model_log)

# serialize model to JSON
model_digit_json = model.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
print("Saved model to disk")