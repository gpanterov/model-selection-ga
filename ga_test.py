import numpy as np
from scipy.optimize import leastsq, minimize
import pandas as pd
import statsmodels.api as sm
import data_tools as tools
from sklearn import svm, linear_model
import time
import random


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
	return len(chrom) - np.sum(chrom)

# Mating function for two chromosomes)
def mate_v1(chrom1, chrom2):
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

def mate_v2(chrom1, chrom2):
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

start = time.time()
# length of chromosome
l = len(train_data.columns)
# Generate a random chromosome
c1 = np.asarray(np.random.randint(0, 2, size=(l, )), dtype=bool)
c2 = np.asarray(np.random.randint(0, 2, size=(l, )), dtype=bool)
for i in range(100000):
	c1 = np.asarray(np.random.randint(0, 2, size=(l, )), dtype=bool)
	c2 = np.asarray(np.random.randint(0, 2, size=(l, )), dtype=bool)
	off1, off2 = mate_v2(c1, c2)

print "It took: ", time.time() - start, " to finish"
# Take variables according to the boolean chromosome
#X = train_data.ix[:, c1]
