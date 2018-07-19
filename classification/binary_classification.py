#!/usr/bin/python3
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import imdb
import numpy as np

def vectorize_data(data, dimension = 10000):
    vectors = np.zeros((len(data), dimension))
    for i, sample in enumerate(data):
        vectors[i, sample] = 1
    return vectors

def graph_train_history(history):
    import matplotlib.pyplot as plt

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
    plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
    plt.plot(epochs, acc, 'ro', label = 'Training accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')

    plt.title('Training and Validation Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

(train_data, train_labels), (test_deta, test_labels) = \
    imdb.load_data(num_words = 10000)

x_train = vectorize_data(train_data)
y_train = np.asarray(train_labels).astype('float32')
x_test = vectorize_data(test_deta)
y_test = np.asarray(test_labels).astype('float32')
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model = Sequential()
model.add(Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', #'rmsprop',
            loss = 'binary_crossentropy',
            metrics = ['accuracy'])
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 10,
                    batch_size = 512,
                    validation_data = (x_val, y_val))

results = model.evaluate(x_test, y_test)
print()
print(results)
graph_train_history(history)