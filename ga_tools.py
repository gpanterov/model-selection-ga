import random
import numpy as np
from bisect import bisect

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

# Sample from a list according to a probability weights
# See http://stackoverflow.com/questions/4113307/pythonic-way-to-select-list-elements-with-different-probability

def cdf(weights):
	total=sum(weights) * 1.
	result=[]
	cumsum=0
	for w in weights:
		cumsum += w 
		result.append(cumsum/total)
	return result

def choice(population, cdf_vals):
	"""
	Returns a random element of population sampled according
	to the weights cdf_vals (produced by the func cdf)
	Inputs
	------
	population: list, a list with objects to be sampled from
	cdf_vals: list/array with cdfs (produced by the func cdf)
	Returns
	-------
	An element from the list population
	"""
	assert len(population) == len(cdf_vals)
	x = random.random()
	idx = bisect(cdf_vals,x)
	return population[idx]

def offspring_pop(parent_pop, fitness, \
					mating_func = mate_point_cross):
	new_pop = []
	cdf_vals = cdf(fitness)
	while len(new_pop) < len(parent_pop):
		chrom1 = choice(parent_pop, cdf_vals)
		chrom2 = choice(parent_pop, cdf_vals)
		off1, off2 = mating_func(chrom1, chrom2)
		new_pop.append(off1)
		new_pop.append(off2)
	return tuple(new_pop)


