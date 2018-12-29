#!/usr/bin/python3
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.datasets import imdb
import numpy as np
from evaluate import cross_validation, held_out_evaluation

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

def build_model():
    model = Sequential()
    model.add(Dense(16, kernal.regularizers.l2(0.001), input_shape = (10000), activation = 'relu' ))
    model.add(Dense(4, kernal.regularizers.l2(0.001), activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer = 'adam', #'rmsprop',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])
    return model

(train_data, train_labels), (test_deta, test_labels) = \
    imdb.load_data(num_words = 10000)

x_train = vectorize_data(train_data)
y_train = np.asarray(train_labels).astype('float32')
x_test = vectorize_data(test_deta)
y_test = np.asarray(test_labels).astype('float32')

#histories, results = cross_validation((x_train, y_train), (x_test, y_test), n_folds = 5, 
#    build_model = build_model,  num_validation_samples = 10000, epochs = 5,
#    batch_size = 512)

history = held_out_evaluation((x_train, y_train), (x_test, y_test), build_model,
    num_validation_samples = 10000, epochs = 5, batch_size = 512)
#graph_train_history(history)
