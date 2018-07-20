import numpy as np
import matplotlib.pyplot as plt

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