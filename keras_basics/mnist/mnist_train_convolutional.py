from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
import numpy as np
from matplotlib import pyplot as plt
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as k

k.common.set_image_data_format('channels_first')

# Criação dos subconjuntos de treino e teste
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Aplicando reshape no conjunto de dados
if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

# Amenizar o uso de memoria (com float 32)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalizamos os valores de acordo com a variação da escala cinza (0 - 255)
X_train /= 255
X_test /= 255


# Como estamos em escala de cinza, definimos a dimensão do pixel como 1
#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Normalizamos os valores de acordo com a variação da escala cinza (0 - 255)
#X_train = X_train / 255
#X_test = X_test / 255

# Aplicamos a solucao de one-hot-coding para classificação multiclasses
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

# Numero de tipos de digitos encontrados no MNIST
num_classes = Y_test.shape[1]

def deeper_cnn_model():
    model = Sequential()

    # A Convolution 2D será a nossa camada de entrada, com 30 mapas de features 
    # de tamanho 5x5 e 'relu' como função de ativação
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))

    # A camada MaxPooling2D será a segunda camada, onde teremos uma amostragem com
    # tamanho 2x2
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Uma nova camada convolucional com 15 mapas de features de dimensões 3x3 e 'relu'
    # como função de ativação
    model.add(Conv2D(15, (3,3), activation = 'relu'))

    # Uma nova subamostragem com um pooling de dimensão 2x2
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Uma camada de DropOut com 20% de probabilidade
    model.add(Dropout(0.2))

    # Uma camada de Flatten preparando os dados para a camada fully connected (final da conv network)
    model.add(Flatten())

    # Camada fully connected de 128 neurônios
    model.add(Dense(128, activation = 'relu'))

    # Seguidamente, outra camada fully connected com 64 neuronios
    model.add(Dense(64, activation = 'relu'))

    # A camada de saida possui o numero de neuronios compativel com o numero de classes a serem classificadas
    # com uma função de ativação do tipo softmax
    model.add(Dense(num_classes, activation = 'softmax', name = 'preds'))

    # Por fim, compilando o modelo
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model

model = deeper_cnn_model()

# O método summary revela quais são as camadas que formam o modelo, seus formatos e o numero
# de parametros envolvidos em cada etapa
model.summary()

# Processo de treinamento do modelo
model_log = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 5, batch_size = 200)

# Avaliação de performance do modelo
scores = model.evaluate(X_test, Y_test, verbose = 0)
print("Erro de: %.2f%%" % (100 - scores[1] * 100))

# Plotando os resultados
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_log.history['accuracy'])
plt.plot(model_log.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

fig

# Salvando o modelo

# serialize model to JSON
model_digit_json = model.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
print("Saved model to disk")