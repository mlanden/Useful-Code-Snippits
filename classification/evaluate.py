
def split_data(train_data, num_validation_samples):
    eval_x, eval_y = train_data
    val_x = eval_x[:num_validation_samples]
    val_y = eval_y[:num_validation_samples]
    train_x = eval_x[num_validation_samples:]
    train_y = eval_y[num_validation_samples:]
    return train_x, train_y, val_x, val_y


def held_out_evaluation(train_data, test_data, build_model, 
    num_validation_samples, epochs, batch_size):
    train_x, train_y, val_x, val_y = split_data(train_data, num_validation_samples)

    model = build_model()
    history = model.fit(train_x, 
                        train_y,
                        epochs = epochs,
                        batch_size = batch_size,
                        validation_data = (val_x, val_y))
    val_info = history.history

    # here id where you tune hyper parameters,
    # Only test model below right before it's ready for production systems
    test_x, test_y = test_data
    test_model = build_model()
    results = model.evaluate(test_x, test_y)
    print(results)

def cross_validation(train_data, test_data, n_folds, build_model, 
    num_validation_samples, epochs, batch_size):
    train_x, train_y = train_data
    test_x, test_y = test_data

    histories = []
    from sklearn.model_selection import StratifiedKFold 
    skf = StratifiedKFold(n_splits = n_folds)
    fold = 0
    for train_idx, val_idx in skf.split(train_x, train_y):
        fold += 1
        print("Beginning fold", fold)
        train_x_fold = train_x[train_idx]
        train_y_fold = train_y[train_idx]
        val_x = train_x[val_idx]
        val_y = train_y[val_idx]

        model = build_model()
        history = model.fit(train_x_fold,
                  train_y_fold,
                  epochs = epochs,
                  batch_size = batch_size,
                  validation_data = (val_x, val_y))
        histories.append(history.history)

    model = build_model()
    model.fit(train_x,
              train_y,
              epochs = epochs,
              batch_size = batch_size)
    results = model.evaluate(test_x, test_y)
    return histories, results










