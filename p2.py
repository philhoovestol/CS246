from pyspark import SparkConf, SparkContext
import matplotlib.pyplot as plt

conf = SparkConf()
sc = SparkContext(conf=conf)

#initialize clusters
dims = 58
centroids = []

centroids_file = open('../../CS246/hw2/p2/sample.txt')
for line in centroids_file.readlines():
	centroid = tuple(line.split(' '))
	centroids.append(centroid)

#euclidean
def euclidean_distance(a, b):
	total = 0
	for d in range(dims):
		total += (float(a[d]) - float(b[d])) ** 2
	return total ** 0.5

#manhattan
def manhattan_distance(a, b):
	total = 0
	for d in range(dims):
		total += abs(float(a[d]) - float(b[d]))
	return total

#return closest centroid in centroids variable to point
def determine_centroid(point):
	closest_centroid = centroids[0]
	closest_distance = manhattan_distance(closest_centroid, point)
	for centroid in centroids:
		current_distance = manhattan_distance(point, centroid)
		if current_distance < closest_distance:
			closest_centroid = centroid
			closest_distance = current_distance
	return closest_centroid

points_lines = sc.textFile('../../CS246/hw2/p2/data.txt')
points = points_lines.map(lambda line: line.split(' ')) 
max_iter = 20

#determine centroid closest to point then return calculated euclidean cost
def calculate_euclidean_cost(point):
	total = 0
	centroid = determine_centroid(point)
	for d in range(dims):
		total += (float(point[d]) - float(centroid[d])) ** 2
	return total

def calculate_manhattan_cost(point):
	total = 0
	centroid = determine_centroid(point)
	for d in range(dims):
		total += abs(float(point[d]) - float(centroid[d]))
	return total

#returns calculated centroid based on cluster object's given points
def calculate_centroid(points):
	result = [0]*dims
	for d in range(dims):
		for point in points:
			result[d] += float(point[d])
		result[d] = result[d] / len(points)
	return result

iter_costs = []
for iter_count in range(max_iter):
	#calculate cost for each point then sum them up
	ind_point_costs = points.map(lambda point: (1, calculate_manhattan_cost(point)))
	iter_costs.append(ind_point_costs.reduceByKey(lambda x1, x2: x1+x2).take(1)[0][1])
	clustered_points = points.map(lambda point: (tuple(determine_centroid(point)), point))
	#recompute centroids and replace centroids with updated version
	centroids = clustered_points.groupByKey().map(lambda cluster: calculate_centroid(cluster[1])).take(10)

for i in range(len(iter_costs)):
	print 'iteration ' + str(i+1) + ' cost ' + str(iter_costs[i])

plt.plot(range(1, max_iter+1),iter_costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Iteration vs Cost with Initial Centroids Randomly Assigned')
plt.show()