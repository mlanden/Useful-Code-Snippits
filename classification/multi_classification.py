#!/usr/bin/python3
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import reuters
from keras.utils import to_categorical
import numpy as np
from utils import vectorize_data, graph_train_history

(train_data, train_labels), (test_data, test_labels) = \
    reuters.load_data(num_words = 10000)


x_train = vectorize_data(train_data)
x_test = vectorize_data(test_data)

#option 1
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

#option 2
# y_train = np.asarray(train_labels)
# y_test = np.asarray(test_labels)

x_val = x_train[:1000]
y_val = y_train[:1000]
partial_x_train = x_train[1000:]
partial_y_train = y_train[1000:]

model = Sequential()
model.add(Dense(64, activation = 'relu', input_shape = (10000,)))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',
              #option 1
              loss = 'categorical_crossentropy',

              #option 2
              # loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 7,
                    batch_size = 512,
                    validation_data = (x_val, y_val))
result = model.evaluate(x_test, y_test)
print(result)
# graph_train_history(history)