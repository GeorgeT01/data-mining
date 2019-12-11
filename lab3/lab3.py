''''
[1] the website Scikit-learn  Mode of access: https://scikit‐learn.org/
[2] Keras library documentation:https://keras.io/
[3] Python site  Mode of access: https://www.python.org/
[4] Library documentation for plotting Matplotlip : https://matplotlib.org/
'''


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.io
from keras_tqdm import TQDMNotebookCallback


data = scipy.io.loadmat('data/digits.mat')

x = np.array(data['X'])
y = np.squeeze(data['y'])

np.place(y, y == 10, 0)
m, n = x.shape

num_labels = 10
input_layer_size = 400
hidden_layer_size = 25

print(f'x shape: {x.shape}\ny size:\t{y.size}')



subplots = 64
draw_seed = np.random.randint(low=0, high=m, size=subplots)
draw_rows = x[draw_seed]
fig, ax = plt.subplots(8, 8, figsize=(8, 8))
for i, axi in enumerate(ax.flat):
    data = np.reshape(draw_rows[i], (20, 20), order='F')
    axi.imshow(data, cmap='binary')
    axi.set(xticks=[], yticks=[])

plt.show()



from keras.models import Sequential
from keras.layers import Dense



model = Sequential()
model.add(Dense(25, activation='sigmoid', input_shape=(400,)))
model.add(Dense(10, activation='sigmoid'))


# ### 3. validation & train sets

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

y = y.reshape(-1, 1)

encoder = OneHotEncoder(sparse=False, categories='auto')
y_onehot = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(x, y_onehot, test_size=0.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ### 4. setup & train

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=1000,
    validation_data=(X_test, y_test),
    verbose=0,
    callbacks=[TQDMNotebookCallback()]
)


[test_cost, test_acc] = model.evaluate(X_test, y_test)
print(f'Results:\nCost:\t\t{test_cost:.2f}\nAccuracy:\t{test_acc*100:.2f}%')


#  ### 5. train result visualization

LABEL_BY_ATTR = {
    'loss': 'Cost',
    'accuracy': 'Accuracy'
}

def plot_history(attr):
    label = LABEL_BY_ATTR[attr]
    plt.figure(figsize=(8, 6))
    plt.plot(history.history[attr], 'r', linewidth=3.0)
    plt.plot(history.history[f'val_{attr}'], 'b' ,linewidth=3.0)
    plt.legend([f'Training {label}', f'Validation {label}'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel(label, fontsize=16)
    plt.title(f'{label} Curves', fontsize=16)

plot_history('loss')


print(f'Accuracy:\t{test_acc*100:.2f}%\nCost:\t\t{test_cost:.2f}')


