import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#importando o mnist (letras)
from keras.datasets import mnist
#importando para visualizar a sequencia de camadas do modelo
from keras.models import Sequential
# modelo simples, importando camadas densas
from keras.layers import Dense
#modulo do keras para as rotinas de pre-processamento
from keras.utils import np_utils
# dropout technique
from keras.layers import Dropout
import keras.optimizers

#divido entre treino e teste
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#pegando o tamanho da rede (numero de pixels)
num_pixels = X_train.shape[1] * X_train.shape[2]  # 28 x 28

# amenizar o uso de memoria (com float 32)
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalizando os valores entre 0 a 1 (ja que as cores vao de 0 a 255)
X_train = X_train / 255
X_test = X_test / 255

# como estamos trabalhando com um problema de classificação multiclasses
# (por ter varios digitos) a gente representa em categorias usando a metodologia
# de one-hot-encoding usando a função to_categorical
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Numero de tipos de digitos encontrados no MNIST
num_classes = y_test.shape[1]

#definindo pixels das proximas camadas
second_layer = int(num_pixels/4) #pixels da segunda camada
third_layer = int(num_pixels/8) #pixels da terceira camada
fourth_layer = int(num_pixels/8) #pixels da quarta camada
fifth_layer = int(num_pixels/16) #pixels da quinta camada


#modelo basico de uma camada

def base_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim = num_pixels, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dropout(0.2)) # 20% de dropout
    model.add(Dense(second_layer, input_dim = num_pixels, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(third_layer, input_dim = num_pixels, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(fourth_layer, input_dim = num_pixels, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(fifth_layer, input_dim = num_pixels, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(num_classes, kernel_initializer = 'normal', activation = 'softmax', name = 'preds'))
    adam = keras.optimizers.Adam(lr = 0.01, decay = 1e-6) # learning rate de 0.01 e decaimento de 1e-6
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

model = base_model()

# revela as camadas que formam o modelo, seus formatos e parametros envolvidos em cada etapa
model.summary()

# processo de treinamento
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 5, batch_size = 100, verbose = 2)

# avaliação a performance
scores = model.evaluate(X_test, y_test, verbose = 0)
print("Erro de: %.2f%%" % (100 - scores[1]*100))