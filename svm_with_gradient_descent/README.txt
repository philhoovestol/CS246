Implementation of soft margin SVM using different gradient descent methods (batch gradient descent, stochastic gradient descent, and mini batch gradient descent).

Contains the following files:

1. features.txt : Each line contains features (comma-separated values) for a single
datapoint. It has 6414 datapoints (rows) and 122 features (columns).
2. target.txt : Each line contains the target variable (y = -1 or 1) for the corresponding
row in features.txt.
3. p1.py : Implementation file. Runs all three methods of gradient descent on the data, iterating until convergence is reached. Creates a graph in results.png that plots the value of the cost function vs. the number of iterations. For more details on the methods see problem 1 in the following handout linked here: http://web.stanford.edu/class/cs246/homeworks/hw4/hw4.pdf
