#-------------------------------------------------------------------------
# AUTHOR: Elijah Chan
# FILENAME: perceptron.py
# SPECIFICATION: build a single-layer and multi-layer perceptron for handwritten digit classification
# FOR: CS 4210- Assignment #3
# TIME SPENT: ~ 4 hrs
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

# importing some libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd


# n is learnign rate ; r is shuffle
n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

# read datasets
df = pd.read_csv('optdigits.tra', sep=',', header=None)
X_training = np.array(df.values)[:, :64].astype(float)
y_training = np.array(df.values)[:, -1].astype(int)

df = pd.read_csv('optdigits.tes', sep=',', header=None)
X_test = np.array(df.values)[:, :64].astype(float)
y_test = np.array(df.values)[:, -1].astype(int)

#trackers
best_acc_perceptron = 0.0
best_params_perceptron = None

best_acc_mlp = 0.0
best_params_mlp = None

total_runs = 0
for lr in n:
    for shuffle in r:
        # iterate over algorithms
        for algo in ['Perceptron', 'MLP']:
            total_runs += 1

            if algo == 'Perceptron':
                #single-layer perceptron
                # eta0 = learning rate, shuffle = shuffle training data
                clf = Perceptron(eta0=lr, shuffle=shuffle, max_iter=1000, random_state=0)
            else:
                # multi-layer perceptron (1 hidden layer with 25 neurons; logistic activation)
                clf = MLPClassifier(activation='logistic',
                                    learning_rate_init=lr,
                                    hidden_layer_sizes=(25,),
                                    shuffle=shuffle,
                                    max_iter=1000,
                                    random_state=0)

            # fit classifier
            clf.fit(X_training, y_training)

            # calc accuracy on test 
            correct = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                pred = clf.predict([x_testSample])[0]
                if pred == y_testSample:
                    correct += 1
            accuracy = correct / len(y_test)

            # check/update bests
            if algo == 'Perceptron':
                if accuracy > best_acc_perceptron:
                    best_acc_perceptron = accuracy
                    best_params_perceptron = (lr, shuffle)
                    print(f"Highest Perceptron accuracy so far: {best_acc_perceptron:.4f}, "
                          f"Parameters: learning rate={lr}, shuffle={shuffle}")
            else:
                if accuracy > best_acc_mlp:
                    best_acc_mlp = accuracy
                    best_params_mlp = (lr, shuffle)
                    print(f"Highest MLP accuracy so far: {best_acc_mlp:.4f}, "
                          f"Parameters: learning rate={lr}, shuffle={shuffle}")

# summary
print("\n=== Search complete ===")
print(f"Total runs: {total_runs}")
if best_params_perceptron is not None:
    print(f"Best Perceptron: accuracy={best_acc_perceptron:.4f}, "
          f"learning_rate={best_params_perceptron[0]}, shuffle={best_params_perceptron[1]}")
else:
    print("No Perceptron results found.")

if best_params_mlp is not None:
    print(f"Best MLP: accuracy={best_acc_mlp:.4f}, "
          f"learning_rate={best_params_mlp[0]}, shuffle={best_params_mlp[1]}")
else:
    print("No MLP results found.")
