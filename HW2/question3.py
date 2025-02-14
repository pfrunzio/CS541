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

#creating a validation data set
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
    def __init__(self, learning_rate, epochs, batch_size, regularization_strength):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size  # (mini) batch
        self.regularization_strength = regularization_strength

        self.w = None   #weight vector
        self.b = None   #bias term

    #finish filling out the model

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #initialize weights and biases
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()

        # implement SDG

        #for each epoch
        for epoch in range(self.epochs):
            # randomize the order of examples in the training set
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                #select a mini batch
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                #estimate the gradient on the mini batch
                y_pred = self.predict(X_batch)
                error = y_pred - y_batch
                gradient_weights = np.dot(X_batch.T, error) / X_batch.shape[0]
                gradient_bias = np.mean(error)

                #update weights and biases
                self.w -= self.learning_rate * gradient_weights
                self.b -= self.learning_rate * gradient_bias

            #report status
            if epoch % 100 == 0:
                y_pred = self.predict(X)
                loss = self.mean_squared_error(y, y_pred)
                print(f"Epoch {epoch}: Loss {loss}")


    def predict(self, X):
        #x.T * w + b
        return np.dot(X, self.w) + self.b

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def compute_cost(self, X, y):
        y_pred = self.predict(X)
        mse = self.mean_squared_error(y, y_pred)
        l2_penalty = (self.regularization_strength / 2) * np.sum(self.w ** 2)  #regularize only w, not b
        return mse + l2_penalty


def grid_search(X_train, y_train, X_val, y_val, learning_rates, epochs_list, batch_sizes, reg_strengths):
    best_model = None
    best_cost = float('inf')
    best_params = {}

    for lr in learning_rates:
        for epochs in epochs_list:
            for batch_size in batch_sizes:
                for reg_strength in reg_strengths:
                    print(f"Training with lr={lr}, epochs={epochs}, batch_size={batch_size}, reg_strength={reg_strength}")

                    #train model with these hyperparameters on training data set
                    model = LinearRegressionSDG(learning_rate=lr, epochs=epochs, batch_size=batch_size, regularization_strength=reg_strength)
                    model.fit(X_train, y_train)

                    #test hyperparameters on validation data set
                    val_cost = model.compute_cost(X_val, y_val)
                    print(f"Validation cost: {val_cost}")

                    #keep track of the best model
                    if val_cost < best_cost:
                        best_cost = val_cost
                        best_model = model
                        best_params = {
                            'learning_rate': lr,
                            'epochs': epochs,
                            'batch_size': batch_size,
                            'regularization_strength': reg_strength
                        }

    print("\nBest Hyperparameters Found:")
    print(best_params)
    return best_model, best_params

def train_problem3():
    #load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48 * 48))
    y_tr = np.load("age_regression_ytr.npy")
    X_test = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48 * 48))
    y_test = np.load("age_regression_yte.npy")

    #split training data into training and validation sets
    X_train, y_train, X_val, y_val = split_data(X_tr, y_tr, 0.2)

    #pick possible hyperparameters
    learning_rates = [0.001, 0.0001, 0.00001, 0.000001]
    epochs_list = [25, 50, 100, 200]
    batch_sizes = [5, 10, 20, 40]
    reg_strengths = [0.01, 0.1, 1, 10]

    #grid search over hyperparameters
    best_model, best_params = grid_search(X_train, y_train, X_val, y_val, learning_rates, epochs_list, batch_sizes, reg_strengths)

    #evaluate on test set
    val_cost = best_model.compute_cost(X_test, y_test)

    #report performance
    print(val_cost)


train_problem3()
