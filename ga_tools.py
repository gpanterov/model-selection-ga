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

def get_fitness(population, fit_func):
	fit = []
	for chrom in population:
		fit.append(fit_func(chrom))
	return tuple(fit)

def mutate_chrom_bool(chrom, mutate_prob=0.1):
	if random.random() < mutate_prob:
		new_chrom = np.array(chrom)
		l = len(chrom)
		indx = random.randint(0, l-1)
		new_chrom[indx] = not new_chrom[indx]
		return tuple(new_chrom)
	else:
		return chrom

def mutate_chrom_bool2(chrom, mutate_prob=0.01):
	new_chrom = []	
	for gene in chrom:
		if random.random() > 1 - mutate_prob: 
			# mutate
			new_chrom.append(not gene)
		else:
			new_chrom.append(gene)
	return tuple(new_chrom)

def offspring_pop(parent_pop, fitness, mutate_func = mutate_chrom_bool,\
					mating_func = mate_point_cross, mutate_prob=0.01):
	new_pop = []
	cdf_vals = cdf(fitness)
	# Keep the top 10 % best solutions from the parents
	indx = np.argsort(fitness)
	cutoff = len(parent_pop) / 10
	best_indx = indx[-cutoff:]
	for i in best_indx:
		new_pop.append(parent_pop[i])
	
	while len(new_pop) < len(parent_pop):
		chrom1 = choice(parent_pop, cdf_vals)
		chrom2 = choice(parent_pop, cdf_vals)
		off1, off2 = mating_func(chrom1, chrom2)
		off1 = mutate_func(off1, mutate_prob)
		off2 = mutate_func(off2, mutate_prob)
		new_pop.append(off1)
		new_pop.append(off2)
	return tuple(new_pop)

def make_chrom_valid(chrom, start_max_length=15):
	above_max = np.sum(chrom) - start_max_length
	if above_max > 0:
		c = np.array(chrom)
		indx = np.where(c == True)[0]
		np.random.shuffle(indx)
		indx_4_ch = indx[0: above_max]
		c[indx_4_ch] = False
		return tuple(c)
	else:
		return chrom
		
