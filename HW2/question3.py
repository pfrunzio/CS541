# linear regression via SGD

#hw1 version of linear regression
import numpy as np

def linear_regression(X_tr, y_tr):
    return np.linalg.inv(X_tr.T @ X_tr) @ X_tr.T @ y_tr

def train_age_regressor():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48 * 48))
    y_tr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48 * 48))
    y_te = np.load("age_regression_yte.npy")

    # Report fMSE cost on the training and testing data (separately)
    def fMSE(X, Y, W):
        return (1/(2.0)) * np.mean(((X@W) - Y)**2)
    w = linear_regression(X_tr, y_tr)
    # Report fMSE cost on the training and testing data (separately)
    # Compute Mean Squared Error (MSE)

    print("Loss on training set is " + str(fMSE(X_tr, y_tr, w)))
    print("Loss on testing set is " + str(fMSE(X_te, y_te, w)))

# train_age_regressor()


#hw2 version

def split_data(X, y, validation_ratio):
    m = X.shape[0]
    validation_size = int(m * validation_ratio)
    indices = np.random.permutation(m)

    X_train = X[indices[:-validation_size]]
    y_train = y[indices[:-validation_size]]
    X_val = X[indices[-validation_size:]]
    y_val = y[indices[-validation_size:]]

    return X_train, y_train, X_val, y_val


class LinearRegressionSDG:
    def __init__(self, batch_size=32, learning_rate=0.01, epochs=1000, regularization_strength=0.1):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization_strength = regularization_strength

        self.w = None   #weight vector
        self.b = None   #bias term

    #finish filling out the model

    def fit(self, X, y):
        m, n = X.shape
        #initialize weights and biases

        #implement SDG
        for epoch in range(self.epochs):

            for i in range(0, m, self.batch_size):

                #compute gradients

                #update weights and biases

    def predict(self, X):
        #x.T * w + b
        return np.dot(X, self.w) + self.b

    # def compute_cost(self, X, y):


def grid_search(X_train, y_train, X_val, y_val, batch_sizes, learning_rates, epochs_list, reg_strengths):
    best_model = None
    best_cost = float('inf')
    best_params = {}

    for batch_size in batch_sizes:
        for lr in learning_rates:
            for epochs in epochs_list:
                for reg_strength in reg_strengths:
                    print(f"Training with batch_size={batch_size}, lr={lr}, epochs={epochs}, reg_strength={reg_strength}")

                    model = LinearRegressionSDG(batch_size=batch_size, learning_rate=lr, epochs=epochs, regularization_strength=reg_strength)
                    model.fit(X_train, y_train)

                    val_cost = model.compute_cost(X_val, y_val)
                    print(f"Validation cost: {val_cost}")

                    #keep track of the best model
                    if val_cost < best_cost:
                        best_cost = val_cost
                        best_model = model
                        best_params = {
                            'batch_size': batch_size,
                            'learning_rate': lr,
                            'epochs': epochs,
                            'regularization_strength': reg_strength
                        }

    print("\nBest Hyperparameters Found:")
    print(best_params)
    return best_model, best_params

def train_problem3():
    #load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48 * 48))
    y_tr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48 * 48))
    y_te = np.load("age_regression_yte.npy")

    #split training data into training and validation sets
    X_train, y_train, X_val, y_val = split_data(X_tr, y_tr, 0.2)

    #pick possible hyperparameters
    batch_sizes = [16, 32, 64, 128]
    learning_rates = [0.001, 0.01, 0.1, 1]
    epochs_list = [500, 1000, 2000, 5000]
    reg_strengths = [0.01, 0.1, 1, 10]

    #grid search over hyperparameters
    grid_search(X_train, y_train, X_val, y_val, batch_sizes, learning_rates, epochs_list, reg_strengths)

    #evaluate on test set

    #report performance



