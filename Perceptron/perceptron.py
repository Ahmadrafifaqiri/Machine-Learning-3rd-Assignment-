import pandas as pd
import numpy as np

def standard_peceptron(X, y, T, lr):
    num_samples, num_features = X.shape
    weight_vector = np.zeros(num_features)
    indices = np.arange(num_samples)
    for t in range(T):
        np.random.shuffle(indices)
        X = X[indices, :]
        y = y[indices]
        for i in range(num_samples):
            predicted = np.sum(weight_vector * X[i])
            if predicted * y[i] <= 0:
                weight_vector = weight_vector + lr * y[i] * X[i]
    return weight_vector

def voted_perceptron(X, y, T, lr):
    num_samples, num_features = X.shape
    weight_vector = np.zeros(num_features)
    list_weight_vector = np.array([])
    list_votes = np.array([])
    indices = np.arange(num_samples)
    votes = 0
    for t in range(T):
        np.random.shuffle(indices)
        X = X[indices, :]
        y = y[indices]
        for i in range(num_samples):
            predicted = np.sum(weight_vector * X[i])
            if predicted * y[i] <= 0:
                list_weight_vector = np.append(list_weight_vector, weight_vector)
                list_votes = np.append(list_votes, votes)
                weight_vector = weight_vector + lr * y[i] * X[i]
                votes = 1
            else:
                votes += 1
    list_weight_vector = np.reshape(list_weight_vector, (list_votes.shape[0], -1))
    return list_weight_vector, list_votes

def average_perceptron(X, y, T, lr):
    num_samples, num_features = X.shape
    weight_vector = np.zeros(num_features)
    average_vector = np.zeros(num_features)
    indices = np.arange(num_samples)
    for t in range(T):
        np.random.shuffle(indices)
        X = X[indices, :]
        y = y[indices]
        for i in range(num_samples):
            predicted = np.sum(weight_vector * X[i])
            if predicted * y[i] <= 0:
                weight_vector = weight_vector + lr * y[i] * X[i]
            average_vector = average_vector + weight_vector
    return average_vector

def standard_evaluate(X, y, weights):
    weights = np.reshape(weights, (-1,1))
    prediction = np.matmul(X, weights)
    prediction[prediction>0] = 1
    prediction[prediction<=0] = -1
    error = np.sum(np.abs(prediction - np.reshape(y,(-1,1)))) / 2 / len(y)
    return error

def voted_evaluate(X, y, weight_vectors, vote_counts):
    vote_counts = np.reshape(vote_counts, (-1,1))
    weight_vectors = np.transpose(weight_vectors)
    predictions = np.matmul(X,weight_vectors)
    predictions[predictions>0] = 1
    predictions[predictions<=0] = -1
    voted_predictions = np.matmul(predictions, vote_counts)
    voted_predictions[voted_predictions>0] = 1
    voted_predictions[voted_predictions<=0] = -1
    error = np.sum(np.abs(voted_predictions - np.reshape(y,(-1,1)))) / 2 / len(y)
    return error
        

if __name__ == "__main__":
    df_train = pd.read_csv("/Users/u1503285/CS-6350-ML/Perceptron/bank-note/train.csv", header=None)
    values = df_train.values
    num_columns = values.shape[1]
    df_train_X = np.copy(values)
    df_train_X[:,num_columns-1] = 1
    df_train_y = values[:,num_columns-1]
    df_train_y = 2 * df_train_y -1

    df_test = pd.read_csv("//Users/u1503285/CS-6350-ML/Perceptron/bank-note/test.csv", header=None)
    values = df_test.values
    num_columns = values.shape[1]  
    df_test_X = np.copy(values)
    df_test_X[:,num_columns-1] = 1
    df_test_y = values[:,num_columns-1]
    df_test_y = 2 * df_test_y -1

    print("Standard Perceptron:\n")
    std_weights = standard_peceptron(df_train_X, df_train_y, 10, 0.01)
    std_error = standard_evaluate(df_test_X, df_test_y, std_weights)
    print("The weight vector: ", std_weights)
    print("Error: ", std_error)

    print("\n\nVoted perceptron:\n")
    voted_weights, vote_vector = voted_perceptron(df_train_X, df_train_y, 10, 0.01)
    voted_error = voted_evaluate(df_test_X, df_test_y, voted_weights, vote_vector)
    print("The weight vector: ", voted_weights)
    print("\nThe count vector: ", vote_vector)
    print("\nError: ", voted_error)

    print("Average perceptron:\n")
    average_weights = average_perceptron(df_train_X, df_train_y, 10, 0.01)
    average_error = standard_evaluate(df_test_X, df_test_y, average_weights)
    print("\nThe weight vector: ", average_weights)
    print("\nError: ", average_error)
