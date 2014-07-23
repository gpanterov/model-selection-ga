import numpy as np
from scipy.optimize import leastsq, minimize
import pandas as pd
import statsmodels.api as sm
import ga_tools as ga
import data_tools as Data
from sklearn import svm, linear_model
import time
import random
reload(ga)
reload(Data)

K = 10
N = 1000
f = Data.fform

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

start = time.time()
# Test loss function
def fit_func(chrom):
	return 1. * np.sum(chrom) / len(chrom)


pop_size = 100
num_gens = 500

# length of chromosome
l = len(train_data.columns)

# Create population
pop = []
for i in range(pop_size):
	chrom = np.asarray(np.random.randint(0, 2, size=(l, )), dtype=bool)
	pop.append(tuple(chrom))
pop = tuple(pop)

# Estimate fit of population

for i in range(num_gens):
	fit = ga.get_fitness(pop, fit_func)
	pop = ga.offspring_pop(pop, fit)
	if max(fit)==1:	
		print "Found maximum at ", i, "th iteration"
		break
print "It took: ", time.time() - start, " to finish"
# Take variables according to the boolean chromosome
#X = train_data.ix[:, c1]
