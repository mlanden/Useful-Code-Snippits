import sys
import logging
from sklearn.svm import SVC
from sklearn import metrics
import multiprocessing as mp


log = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

""" Use this like this:
    predictions, model_params = mux_model_predict(parameters, x, y)
    where parameters['model'] has the string name of the model

"""

class Models():
    SVM_MODEL = "Support Vector Machine"
    SVM_RANDMIXED_OPTIMIZE_MODEL = "Randomly Optimized Support vector machine"
    RANDOM_FOREST_MODEL = "Random Forest"
    KNN_MODEL = "KNN Classifier"
    DECISION_TREE_MODEL = "Decision Tree"
    MLP_MODEL = "Multi-Layer Perceptron"


""" Each of the model creation methods return the requested model
    as well as a flag indicating where the model can be 
    trained and predicted with the simple sklearn cross_val predit"""

def create_svm(params, x, y):
    """ creates a SVM with the given parameters """
    clf = SVC(
        C=params['C'],
        gamma=params['gamma'],
        cache_size=params['cache_size'],
        kernel='rbf',
        class_weight='balanced',
        max_iter=params['max_iter'],
    )
    return clf

def create_random_search_svm(params, x, y):
    """Creates a random search over some hyperparameter applied to a svm"""
    from sklearn.model_selection import  GridSearchCV
    from scipy import stats
    from sklearn import preprocessing

    param_dists = {
          'C': [.1, 1, 10, 100, 1000, 10000],
          'gamma': [.01, .1, 1, 10],
    }

    clf = SVC(max_iter=50000, kernel='rbf', class_weight='balanced', cache_size=1000*5)

    grid = GridSearchCV(
          clf,
          param_grid=param_dists,
          cv = 5,
          pre_dispatch=8,
          scoring='f1',
          n_jobs=mp.cpu_count() - 1
    )

    test_scaled = preprocessing.scale(x)

    grid.fit(test_scaled, y)

    clf = SVC(
         C = grid.best_params_['C'],
         gamma=grid.best_params_['gamma'],
         max_iter=50000, kernel='rbf',
         class_weight='balanced',
         cache_size=1000*5
    )
    return grid

def create_random_forest(params, x, y):
    """ creates a random forest classifier"""
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
            n_estimators = params['n_estimators'],
            class_weight = params['class_weight'],
            oob_score=True,
            random_state=1000,
            max_depth = params['max_depth'],
            #min_samples_split=2,
            n_jobs = mp.cpu_count() - 1)
    return clf

def create_knn(params, x, y):
    """ creates a k nearest neihbor classifier """
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(
        n_neighbors = params['num_neighbors'])
    return knn

def create_decision_tree(params, x, y):
    """ creates a decision tree classifier """
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier()
    return tree

def create_multilayer_perceptron(params, x, y):
    """ creates a MLP """
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(
        hidden_layer_sizes = params["hidden_layer_sizes"],
        activation = params["activation"],
        solver = params["solver"],
        alpha = params["alpha"],
        early_stopping = params["early_stopping"]
    )

    return mlp

def non_nn_cv_predict(model, params, x, y):
    from sklearn.model_selection import cross_val_predict
    predictions = cross_val_predict(
        model,
        x,
        y,
        cv = params['n_folds']
    )
    return predictions, model.get_params()

model_lookup = {
    Models.SVM_MODEL : create_svm,
    Models.SVM_RANDMIXED_OPTIMIZE_MODEL : create_random_search_svm,
    Models.RANDOM_FOREST_MODEL : create_random_forest,
    Models.KNN_MODEL : create_knn,
    Models.DECISION_TREE_MODEL : create_decision_tree,
    Models.MLP_MODEL : create_multilayer_perceptron
}

def mux_model_predict(params, x, y):
    """ returns the model connected to the string name """
    log.debug("Using a {} model for classification".format(params['model']))
    model = model_lookup[params['model']](params, x, y)

    return non_nn_cv_predict(model, params, x, y)

def get_model(params):
    log.debug("Using a {} model for classification".format(params['model']))
    model = model_lookup[params['model']](params, x, y )
    return model

