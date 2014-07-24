import numpy as np
import pandas as pd
import ga_tools as ga
import data_tools as Data
import time
import random
import data_models as models

reload(models)
reload(ga)
reload(Data)

np.random.seed(12345)


###############
# Create Data #
###############
K = 10
K2 = 10 * K
N = 5000
f = Data.fform

operators = '+' * (K - 1)
coefs = np.random.normal(0, 2, size=(K, ))
print coefs
y1, train_data1 = Data.create_data('sample_data1.csv', N, K, K2, 
												f, operators, coefs)


data = pd.read_csv('sample_data1.csv', header=None)
y = data.ix[:, 0]
train_data = data.ix[:, 1:]
N, Kall = np.shape(train_data)
train_data.columns = range(Kall)
print "Created the data"
######
# GA #
######

start = time.time()
# Test loss function
def fit_func(chrom):
	return 1. * np.sum(chrom) / len(chrom)

def model_fit(chrom, y, train_data):
	X = train_data.ix[:, list(chrom)]
	adjR2 = models.gof_ols(y, X)
	n = np.sum(chrom)
	b = 0
	c = 5e-4
	penalty = b * n + c * (n ** 2)
	return  adjR2 * np.exp(-penalty)

fit_func2 = lambda chrom: model_fit(chrom, y, train_data)


pop_size = 500
num_gens = 500

# length of chromosome
l = len(train_data.columns)

# Create population
pop = []
for i in range(pop_size):
	chrom = np.asarray(np.random.randint(0, 2, size=(l, )), dtype=bool)
	chrom = ga.make_chrom_valid(chrom)
	pop.append(tuple(chrom))
pop = tuple(pop)
print "Created the pop"

c_best = [False] * l
c_best[0:K] = [True] * K
fit_best_model = fit_func2(c_best)
print "The fitness of the best model is: ", fit_best_model, "\n"
# Estimate fit of population
print "Calculated he best model and starting the loop"
all_best_fit = []

start_time2 = time.time()
for i in range(num_gens):
	fit = ga.get_fitness(pop, fit_func2)
	pop = ga.offspring_pop(pop, fit, 
			mating_func = ga.mate_random_cross, mutate_prob=0.01)

	indx = np.argsort(fit)
	best = pop[indx[-1]]
	best_vars = np.where(np.array(best) == True)

	all_best_fit.append(fit[indx[-1]])
	if len(set(all_best_fit[-30:])) == 1 and len(all_best_fit) > 30:
		print "Keep getting same results so we break"
		break

	if i % 10 == 0:
		print "The best model is ", list(best_vars[0]), \
			 " and its fitness is: ", fit[indx[-1]]
		if fit_best_model <= fit[indx[-1]]:
			print "PASSED FITNESS OF TRUE MODEL"
		print "\n"
		print "To complete 10 iterations it took: ", time.time() - start_time2
		start_time2 = time.time()

print "\n"
print "#" * 20
print "\n"
print "It took: ", time.time() - start, " to finish"
# Take variables according to the boolean chromosome
#X = train_data.ix[:, c1]
