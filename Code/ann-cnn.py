from keras.models import Sequential
from keras.layers import Flatten, BatchNormalization, Dense, Activation, Conv2D, MaxPooling2D
from keras import metrics
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import matplotlib.pyplot as plt
from data import data
from dense_custom_layer import dense_custom_layer
import itertools
from keras.optimizers import SGD
from decimal import Decimal

# Mode: 0 = ANN, 1 = CNN
MODE = 0
EPOCHS = 200
RUNS = 2
LRATES = [1e-1, 1e-2, 1e-3, 1e-4]
BSIZES = [32]

#########
# Model #
#########
# Make the model
def make_model(learning_rate):
    # Create model
    model = Sequential()
    # Flatten layer
    model.add(Flatten())
    # Batch normalization layer
    model.add(BatchNormalization())
    # 4 dense layers (64, 64, 64, 7) including 1 custom layer
    model.add(dense_custom_layer(64))
    model.add(Dense(units=64, activation='relu', input_shape=(77, 66, 3,))) # add input layer
    model.add(Dense(units=64, activation='relu')) # add layer
    model.add(Dense(units=7, activation='sigmoid')) # add output layer
    # Activation layer
    model.add(Activation('softmax'))
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
def train_model(model, x_train, y_train, training_batch_size, EPOCHS, lr, graph = False):
    # Fit the model
    history = model.fit(x_train, y_train, epochs = EPOCHS, batch_size = training_batch_size)
    # Graph the model if specified
    if graph:
        for metric in history.history.keys():
            plt.plot(history.history[metric])
            if MODE == 0:
                plt.title("ANN Testing " + str(metric).capitalize() + " for LR = " + '%.2E' % Decimal(lr))
            else:
                plt.title("CNN Testing " + str(metric).capitalize() + " for LR = " + '%.2E' % Decimal(lr))
            plt.ylabel(str(metric).capitalize())
            plt.xlabel("Epoch")
            plt.show()
    return history

# Evaluate the model
def eval_model(model, x_set, y_set):
    loss_and_metrics = model.evaluate(x_set, y_set, batch_size=128)
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
def plot_training(histories, lr, graph = False):
    # Iterate through each metric in the history
    for metric in histories[0].history.keys():
        total_metric = np.array([np.asarray(histories[0].history[metric])])
        for i in np.arange(1, len(histories)):
            total_metric = np.concatenate((total_metric, np.array([np.asarray(histories[i].history[metric])])), axis = 0)
        avg_metric = np.average(total_metric, axis = 0)
        # Graph is specified
        if graph:
            plt.plot(avg_metric)
            if MODE == 0:
                plt.title("Average ANN " + str(metric).capitalize() + " for LR = " + '%.2E' % Decimal(lr))
            else:
                plt.title("Average CNN " + str(metric).capitalize() + " for LR = " + '%.2E' % Decimal(lr))
            plt.ylabel(str(metric).capitalize())
            plt.xlabel("Epoch")
            plt.legend()
            plt.show()

# Plot all training
def plot_all_training(all_histories):
    metrics = ['acc', 'loss']
    for metric in metrics:
        # Iterate through each history
        count = 0
        for histories in all_histories:
            total_metric = np.array([np.asarray(histories[0].history[metric])])
            for i in np.arange(1, len(histories)):
                total_metric = np.concatenate((total_metric, np.array([np.asarray(histories[i].history[metric])])), axis = 0)
            avg_metric = np.average(total_metric, axis = 0)
            # Plot
            plt.plot(avg_metric, label=str('%.2E' % Decimal(LRATES[count])))
            count += 1
        # Plot
        if MODE == 0:
            plt.title("Average ANN " + str(metric).capitalize())
        else:
            plt.title("Average CNN " + str(metric).capitalize())
        plt.ylabel(str(metric).capitalize())
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

# Average evaluations and accuracies
def avg_evals(evals):
    avg_eval = np.average(evals, axis = 0)
    print("Average loss and accuracy on validation data:", avg_eval)
    return avg_eval[1]

########
# Main #
########
def main():
    print("\n\n#########")
    print("# BEGIN #")
    print("#########\n\n")
    # Read data
    bee_data = data()
    #bee_data = data("../Data/bee_data.csv", "../Data/bee_imgs", 5) # Column 5 = Species
    num_classes = len(np.unique(bee_data.bee_targets))
    # Set hyperparameters
    hyperparameters = list(itertools.product(LRATES, BSIZES))
    # Configure models
    acc_models = np.empty(len(hyperparameters))
    # Histories
    all_histories = np.empty(len(hyperparameters), dtype = object)
    # Find best combination of hyperparameters
    print("Beginning grid search...")
    for i in range(len(hyperparameters)):
        histories = np.empty(RUNS, dtype = object)
        evals = np.empty(RUNS, dtype = object)
        print("\n--------------------------------------------------")
        print("Training model with learning rate = " + '%.2E' % Decimal(hyperparameters[i][0]) + "...")
        print("--------------------------------------------------")
        # Run RUN number of times
        for run in range(RUNS):
            print("\n------------------------------")
            print("Run #" + str(run) + " for LR = " + '%.2E' % Decimal(hyperparameters[i][0]))
            print("------------------------------\n")
            # Run either ANN or CNN
            if MODE == 0:
                model = make_model(hyperparameters[i][0])
            else:
                model = make_conv_model(hyperparameters[i][0])
            # Train the model
            history = train_model(model, bee_data.training_data, one_hot(bee_data.training_targets, num_classes),
                hyperparameters[i][1], EPOCHS, hyperparameters[i][0])
            histories[run] = history
            # Evaluate the model
            print(">> Evaluating model...")
            eval = eval_model(model, bee_data.validation_data, one_hot(bee_data.validation_targets, num_classes))
            evals[run] = np.array(eval)
        # Plot the training performance
        plot_training(histories, LRATES[i])
        # Data collection
        all_histories[i] = histories
        acc_models[i] = avg_evals(evals)
    # Plot histories
    plot_all_training(all_histories)
    # Output best model
    index_max = np.argmax(acc_models)
    print("\n##############################")
    print("Best model from all runs: ", hyperparameters[index_max])
    print("Best model accuracy from all runs: ", acc_models[index_max])
    print("\n##############################\n")
    # Make model with best combination, train with train and valid set, then test
    print("\n--------------------------------------------------")
    print("Training new model with best found hyperparameters...")
    print("--------------------------------------------------\n")
    if MODE == 0:
        model = make_model(hyperparameters[index_max][0])
    else:
        model = make_conv_model(hyperparameters[index_max][0])
    # Train model
    history = train_model(model, bee_data.training_data, one_hot(bee_data.training_targets, num_classes),
        hyperparameters[index_max][1], EPOCHS, hyperparameters[index_max][0], graph = True)
    # Evaluate model
    print(">> Evaluating new model...")
    eval = eval_model(model, bee_data.testing_data, one_hot(bee_data.testing_targets, num_classes))
    # Output test performance
    print(">> Performance on testing data:", eval)

if __name__ == "__main__":
    main()
