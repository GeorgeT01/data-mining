''''
George Alromhin gr.858301
example of deep learning with a library Keras.. https://keras.com/
Documentation library for charting Matplotlib.. https://matplotlib.org/
Documentation of tensorflow https://www.tensorflow.org/install/pip
OF. PIP site .. https://pypi.org/project/pip/





'''


import tensorflow as tf
import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, SimpleRNN
from inspect import getmembers, isfunction
import matplotlib.pyplot as plt
from tensorflow.keras.activations import linear, relu, softplus

#Как задать модель нейронной сети. Какие есть интерфейсы и их параметры? Как задать весовые коэффициенты нейронной сети?

W = np.random.rand(2, 2)
b = np.random.rand(2)

visible = Input(shape=(2,))
hidden = Dense(units=2, weights=[W, b])(visible) # layer with weights

model = Model(inputs=visible, outputs=hidden)


#Как задать полносвязный слой нейронной сети?

Dense(units=32)


#Как задать свёрточный слой нейронной сети?
Conv1D(kernel_size=200, filters=20)


#Какие есть средства для работы с рекуррентными нейросетями?

SimpleRNN(units=32)


#Как задать функцию активации нейронной сети и какие поддерживаются в keras?

Dense(64, activation='tanh')

print('Activation Functions:')
[name for name, obj in getmembers(tf.keras.activations) if isfunction(obj) and name != 'deserialize']

#Чем отличается linear от ReLU, softplus?

x = np.linspace(-10, 10)
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(14, 4))

for i, f in enumerate([linear, relu, softplus]):
    ax[i].plot(x, f(x))
    ax[i].set_title(f.__name__)

plt.show()

#Как задать функцию ошибки / потерь нейронной сети? Как задать метод обучения нейронной сети?

model.compile(loss='mean_squared_error', optimizer='sgd')

#Чем отличается mean_squared_error от cosinus_proxmity, по каким формулам они вычисляются?

#Чем отличается SGD от rprop, Adadelta, Adam; nesterov от momentum?
keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#Чем отличается SGD от rprop, Adadelta, Adam; nesterov от momentum?



#Как указать обучающую выборку?
train_data = np.random.random((2, 2))
model.fit(train_data, epochs=10)

#Как указать обучающую выборку?

train_data = np.random.random((2, 2))
model.fit(train_data, epochs=10)
