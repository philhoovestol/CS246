implementation of iterative k-means clustering using the Spark MapReduce library in python.

There are 4 files:

1. data.txt contains the dataset which has 4601 rows and 58 columns. Each row is a
document represented as a 58 dimensional vector of features. Each component in the
vector represents the importance of a word in the document.
2. c1.txt contains k initial cluster centroids. These centroids were chosen by selecting
k = 10 random points from the input data.
3. c2.txt contains initial cluster centroids which are as far apart as possible. (You can
do this by choosing 1st centroid c1 randomly, and then finding the point c2 that is
farthest from c1, then selecting c3 which is farthest from c1 and c2, and so on).
4. p2.py contains the implementation. Can be set to use either c1.txt or c2.txt as initial cluster centroids and to use either Euclidean or Manhattan distance. Plots the cost of the resulting clustering as a function of the number of iterations.

