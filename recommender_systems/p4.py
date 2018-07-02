import numpy as np

#p is an m*m diagonal matrix, ith element is # of movies user i likes
#q is an n*n diagonal matrix, ith element is # of watches for movie i

#load r
r_raw =[]
f = open('user-shows.txt', 'r')
for line in f.readlines():
	r_raw.append(map(int, line.split(' ')))
r = np.matrix(r_raw)

#derive p and q
m = r.shape[0] # of users
n = r.shape[1] # of movies
p_values = []
q_values = [0]*n
for row in r.A:
	num_user_watches = list(row).count(1)
	p_values.append(num_user_watches)
	for i in range(n):
		q_values[i] += row[i]

def power(my_list):
    return [1/(x**0.5) for x in my_list ]

p_minushalf = np.diag(power(p_values))
q_minushalf = np.diag(power(q_values))

def ithmovie(i):
	f = open('shows.txt', 'r')
	return f.readlines()[i].strip('\n')

def print_alexs_top_5_from_100(tau):
	row_alex_first_100 = list(tau.A[499])[:100]
	row_alex_top_5 = sorted(row_alex_first_100, reverse=True)[:5]
	for sim_score in row_alex_top_5:
		print ithmovie(row_alex_first_100.index(sim_score))+' -> '+str(sim_score)

tau_user_user = p_minushalf.dot(r).dot(np.transpose(r)).dot(p_minushalf).dot(r)
tau_item_item = r.dot(q_minushalf).dot(np.transpose(r)).dot(r).dot(q_minushalf)
print '\nuser-user collaborative filtering results'
print_alexs_top_5_from_100(tau_user_user)
print '\nitem-item collaborative filtering results'
print_alexs_top_5_from_100(tau_item_item)
