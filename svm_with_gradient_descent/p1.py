import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import timeit

d = 122
learning_rate = 0.0000003
epsilon = 0.25
c = 100
n = 6000
x = []
y = []
k = 0
w =[0]*d
b = 0
ks = []

f = open('features.txt', 'r')
for line in f.readlines():
	x.append(map(int, line.split(',')))
f = open('target.txt', 'r')
for line in f.readlines():
	y.append(int(line))

def total_cost():
	total = 0.5
	first_summation_total = 0
	for j in range(d):
		first_summation_total += w[j] ** 2
	total *= first_summation_total
	summation_i_total = 0
	for i in range(n):
		summation_j_total = 0
		for j in range(d):
			summation_j_total += w[j]*x[i][j]
		value = 1 - y[i] * (summation_j_total + b)
		if value > 0:
			summation_i_total += value
	return total + c * summation_i_total

#run bgd
print 'running bgd'

bgd_costs = []

def bgd_convergence():
	if k == 0: return False
	delta = abs(bgd_costs[k-1] - bgd_costs[k]) * 100 / bgd_costs[k-1]
	print 'delta: ' + str(delta)
	return delta < epsilon

def bgd_w_delta(j):
	summation_total = 0
	for i in range(n):
		if y[i]*(np.dot(x[i], w) + b) < 1:
			summation_total -= y[i]*x[i][j]
	return w[j] + c * summation_total

def bgd_b_delta():
	summation_total = 0
	for i in range(n):
		if y[i]*(np.dot(x[i], w) + b) < 1:
			summation_total -= y[i]
	return c * summation_total

bgd_start = timeit.default_timer()
bgd_costs.append(total_cost())
while not bgd_convergence():
	for j in range(d):
		w[j] -= learning_rate * bgd_w_delta(j)
	b -= learning_rate * bgd_b_delta()
	k += 1
	print 'completed ' + str(k) + ' iterations'
	cost = total_cost()
	print 'cost: ' + str(cost)
	bgd_costs.append(cost)
bgd_iterations = k
bgd_time = timeit.default_timer() - bgd_start

#run sgd
print 'running sgd'

sgd_costs = []
w =[0]*d
b = 0
k = 0
i = 1
learning_rate = 0.0001
epsilon = 0.001
previous_delta_cost = 0

def convergence(costs):
	global previous_delta_cost
	if k == 0: return False
	delta_percent = abs(costs[k-1] - costs[k]) * 100 / costs[k-1]
	delta_cost = 0.5 * previous_delta_cost + 0.5 * delta_percent
	previous_delta_cost = delta_cost
	print 'delta: '+str(delta_cost)
	return delta_cost < epsilon

def sgd_w_delta(j):
	lw = 0
	if y[i]*(np.dot(x[i], w) + b) < 1:
		lw -= y[i]*x[i][j]
	return w[j] + c * lw

def sgd_b_delta():
	lb = 0
	if y[i]*(np.dot(x[i], w) + b) < 1:
		lb -= y[i]
	return c * lb

sgd_start = timeit.default_timer()

#shuffle x and y
xy = list(zip(x,y))
shuffle(xy)
x, y = zip(*xy)

sgd_costs.append(total_cost())
while not convergence(sgd_costs):
	for j in range(d):
		w[j] -= learning_rate * sgd_w_delta(j)
	b -= learning_rate * sgd_b_delta()
	i = (i % n) + 1
	k += 1
	cost = total_cost()
	print 'completed ' + str(k) + ' iterations'
	print 'cost: '+str(cost)
	sgd_costs.append(cost)
sgd_iterations = k
sgd_time = timeit.default_timer() - sgd_start

#run mbgd
print 'running mbgd'

mbgd_costs = []
learning_rate = 0.00001
epsilon = 0.01
batch_size = 20
w = [0]*d
b = 0
l = 0
k = 0
previous_delta_cost = 0

def mbgd_w_delta(j):
	summation_total = 0
	for i in range(l * batch_size + 1, min(n,(l + 1)*batch_size)):
		if y[i]*(np.dot(x[i], w) + b) < 1:
			summation_total -= y[i]*x[i][j]
	return w[j] + c * summation_total

def mbgd_b_delta():
	summation_total = 0
	for i in range(l * batch_size + 1, min(n,(l + 1)*batch_size)):
		if y[i]*(np.dot(x[i], w) + b) < 1:
			summation_total -= y[i]
	return c * summation_total

mbgd_start = timeit.default_timer()

#shuffle x and y
xy = list(zip(x,y))
shuffle(xy)
x, y = zip(*xy)

mbgd_costs.append(total_cost())
while not convergence(mbgd_costs):
	for j in range(d):
		w[j] -= learning_rate * mbgd_w_delta(j)
	b -= learning_rate * mbgd_b_delta()
	l = (l + 1) % ((n + batch_size - 1)/batch_size)
	k += 1
	cost = total_cost()
	print 'completed ' + str(k) + ' iterations'
	print 'cost: '+str(cost)
	mbgd_costs.append(total_cost())
mbgd_iterations = k
mbgd_time = timeit.default_timer() - mbgd_start

print 'bgd completed in '+str(bgd_iterations)+' iterations in '+str(bgd_time)+' seconds'
print 'sgd '+str(sgd_iterations)+ ' in ' + str(sgd_time)
print 'mbgd in '+str(mbgd_iterations)+' in '+str(mbgd_time)

plt.plot(bgd_costs)
plt.plot(sgd_costs)
plt.plot(mbgd_costs)
plt.legend(['Batch Gradient Descent', 'Stochastic Gradient Descent','Mini Batch Gradient Descent'])
plt.xlabel('k')
plt.ylabel('cost')
plt.show()
