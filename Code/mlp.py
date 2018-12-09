from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import metrics
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import matplotlib.pyplot as plt
from data import data
import itertools
from keras.optimizers import SGD

#########
# Model #
#########
# Make the model
def make_model(learning_rate):
    # Create model
    model = Sequential()
    # Flatten layer
    model.add(Flatten())
    # 4 dense layers (64, 64, 64, 7)
    model.add(Dense(units=64, activation='relu', input_shape=(77, 66, 3,))) # add input layer
    model.add(Dense(units=64, activation='relu')) # add layer
    model.add(Dense(units=64, activation='relu')) # add layer
    model.add(Dense(units=7, activation='sigmoid')) # add output layer
    # Compile
    opt = SGD(lr = learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Make convolutional model
def make_conv_model(learning_rate):
    # Create model
    model = Sequential()
    # First layer of convolution
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(77, 66, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    # Second layer of convolution
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(77, 66, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten
    model.add(Flatten())
    # 3 Dense layers (64, 64, 7)
    model.add(Dense(units=64, activation='relu')) # add layer
    model.add(Dense(units=64, activation='relu')) # add layer
    model.add(Dense(units=7, activation='sigmoid')) # add output layer
    # Compile
    opt = SGD(lr = learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Train the model
def train_model(model, x_train, y_train, training_batch_size, training_epochs, graph = False):
    # Fit the model
    history = model.fit(x_train, y_train, epochs = training_epochs, batch_size = training_batch_size)
    # Graph the model if specified
    if graph:
        for metric in history.history.keys():
            plt.plot(history.history[metric])
            plt.title("Model " + str(metric))
            plt.ylabel(metric)
            plt.xlabel("Epoch")
            plt.show()
    return history

# Evaluate the model
def eval_model(model, x_set, y_set):
    loss_and_metrics = model.evaluate(x_set, y_set, batch_size=128)
    # classes = model.predict(x_set, batch_size=128)
    print("eval_model set metrics:\n", loss_and_metrics)
    return loss_and_metrics

 # One hot encoding
def one_hot(targets, num_classes):
    return keras.utils.to_categorical(targets, num_classes = num_classes)

# Class analysis
def class_composition(targets):
    unique, counts = np.unique(targets, return_counts=True)
    counts = counts / float(len(targets))
    print(dict(zip(unique, counts)))

#################
# Visualization #
#################
# Plot one training
def plot_training(histories, graph = False):
    # Iterate through each metric in the history
    for metric in histories[0].history.keys():
        total_metric = np.array([np.asarray(histories[0].history[metric])])
        for i in np.arange(1, len(histories)):
            total_metric = np.concatenate((total_metric, np.array([np.asarray(histories[i].history[metric])])), axis = 0)
        ave_metric = np.average(total_metric, axis = 0)
        # Graph is specified
        if graph:
            plt.plot(ave_metric)
            plt.title("Average Model " + str(metric))
            plt.ylabel(metric)
            plt.xlabel("Epoch")
            plt.show()

# Plot all training
def plot_all_training(all_histories):
    # Measure accuracy
    metric = 'acc'
    # Iterate through each history
    for histories in all_histories:
        total_metric = np.array([np.asarray(histories[0].history[metric])])
        for i in np.arange(1, len(histories)):
            total_metric = np.concatenate((total_metric, np.array([np.asarray(histories[i].history[metric])])), axis = 0)
        ave_metric = np.average(total_metric, axis = 0)
        # Plot
        plt.plot(ave_metric)
    # Plot
    plt.title("Average Model " + str(metric))
    plt.ylabel(metric)
    plt.xlabel("Epoch")
    plt.show()

# Average evaluations and accuracies
def ave_evals(evals):
    # Populate a
    # total_eval = np.array([np.asarray(evals[0])])
    # for i in np.arange(1, len(evals)):
    #     total_eval = np.concatenate((total_eval, np.array([np.asarray(evals[i])])))
    # total_eval = np.array(evals)

    # Pretty sure that the code above can be replaced with this line below
    # Keeping code above until we're sure
    ave_eval = np.average(total_eval, axis = 0)
    print("Average Loss and Accuracy: ", ave_eval)
    return ave_eval[1]

########
# Main #
########
def main():
    # Read data
    bee_data = data("../Data/bee_data.csv", 5, "../Data/resized_bee_imgs") # Column 5 = Species
    num_classes = len(np.unique(bee_data.bee_targets))
    # Set hyperparameters
    training_epochs = 100
    runs = 3
    learning_rates = [1e-2, 1e-6]
    training_batch_sizes = [32]
    hyperparameters = list(itertools.product(learning_rates, training_batch_sizes))
    # Configure models
    acc_models = np.empty(len(hyperparameters))
    # Histories
    all_histories = np.empty(len(hyperparameters), dtype = object)
    # Find best combination of hyperparameters
    for i in range(len(hyperparameters)):
        histories = np.empty(runs, dtype = object)
        evals = np.empty(runs, dtype = object)
        for run in range(runs):
            model = make_conv_model(hyperparameters[i][0])
            history = train_model(model, bee_data.training_data,
                one_hot(bee_data.training_targets, num_classes),
                hyperparameters[i][1], training_epochs)
            eval = eval_model(model, bee_data.validation_data,
                one_hot(bee_data.validation_targets, num_classes))
            histories[run] = history
            evals[run] = eval
        plot_training(histories)
        all_histories[i] = histories
        acc_models[i] = ave_evals(evals)
    # Plot histories
    print("Display learning curves over hyperparameter sets")
    plot_all_training(all_histories)
    # Output best model
    index_max = np.argmax(acc_models)
    print("Best model: ", hyperparameters[index_max])
    print("Best model acc: ", acc_models[index_max])
    # Make model with best combination, train with train and valid set, then test
    training_epochs = 100
    model = make_conv_model(hyperparameters[index_max][0])
    history = train_model(model, bee_data.training_data,
        one_hot(bee_data.training_targets, num_classes), hyperparameters[index_max][1],
        training_epochs, graph = True)
    eval = eval_model(model, bee_data.testing_data, one_hot(bee_data.testing_targets, num_classes))
    # Output test performance
    print("Evaluation on Test: ", eval)

if __name__ == "__main__":
    main()
