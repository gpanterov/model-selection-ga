import numpy as np
from scipy.optimize import leastsq, minimize
import pandas as pd
import statsmodels.api as sm
import data_tools as tools
from sklearn import svm, linear_model
import time
import random
from bisect import bisect
reload(tools)



K = 10
N = 1000
f = tools.fform

operators = '+*+**+***'


###############
# Create Data #
###############
coefs = np.random.normal(size=(K, ))
train_data = np.random.normal(size=(N, K))

noise = np.random.normal( size=(N, ))
y = f(coefs, train_data, operators) + noise

# Crate other (non-relevant) data
other_data = np.random.normal(size=(N,K))

# Create out-of-sample data
new_data = np.random.normal(size=(N / 3, K))
new_noise = np.random.normal(size=(N / 3, ))
new_y = f(coefs, new_data, operators) + new_noise

train_data = pd.DataFrame(np.column_stack((train_data, other_data)))
y = pd.Series(y)

######
# GA #
######

# Test loss function
def loss_func(chrom):
	return 1. * len(chrom) - np.sum(chrom)

# Mating function with crossover point
def mate_point_cross(chrom1, chrom2):
	l = len(chrom1)

	offspring1 = [True] * l
	crossover_point = np.random.randint(1, l-1) + 1

	offspring1[0: crossover_point] = chrom1[0: crossover_point]
	offspring1[crossover_point:] = chrom2[crossover_point :]

	offspring2 = [True] * l
	crossover_point = np.random.randint(1, l-1) + 1

	offspring2[0: crossover_point] = chrom1[0: crossover_point]
	offspring2[crossover_point:] = chrom2[crossover_point :]

	return tuple(offspring1), tuple(offspring2)

# mating function with random crossover
def mate_random_cross(chrom1, chrom2):
	l = len(chrom1)

	offspring1 = [True] * l
	offspring2 = [True] * l

	for i in range(l):
		if random.getrandbits(1) == 0:
			offspring1[i] = chrom1[i]
		else:
			offspring1[i] = chrom2[i]

		if random.getrandbits(1) == 0:
			offspring2[i] = chrom1[i]
		else:
			offspring2[i] = chrom2[i]
	return tuple(offspring1), tuple(offspring2)

def cdf(weights):
	total=sum(weights)
	result=[]
	cumsum=0
	for w in weights:
		cumsum += w * 1.
		result.append(cumsum/total)
	return result

def choice(population, cdf_vals):
	assert len(population) == len(cdf_vals)
	x = random.random()
	idx = bisect(cdf_vals,x)
	return population[idx]

start = time.time()
# length of chromosome
l = len(train_data.columns)
# Generate a random chromosome
c1 = np.asarray(np.random.randint(0, 2, size=(l, )), dtype=bool)
c2 = np.asarray(np.random.randint(0, 2, size=(l, )), dtype=bool)
pop = []
for i in range(4):
	pop.append(np.asarray(np.random.randint(0, 2, size=(l, )), dtype=bool))

fit = []
for gene in pop:
	fit.append(loss_func(gene))

a=['A', 'B', 'C']
w =[50., 1., 30.]
cdf_vals = cdf(w)
res=[]
for i in range(1000):
	res.append(choice(a, cdf_vals))
res = np.array(res)
print "There are: ", np.sum(res=='A'), ' As'
print "There are: ", np.sum(res=='B'), ' Bs'
print "It took: ", time.time() - start, " to finish"
# Take variables according to the boolean chromosome
#X = train_data.ix[:, c1]
