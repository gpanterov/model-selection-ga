import pandas as pd
import ga_tools as ga
import data_tools as Data
import random
import data_models as models
import numpy as np
import sys
import boto
import aws_tools as aws_tools
import statsmodels.api as sm

#inFile = sys.argv[1]
output_key_name = sys.argv[2]
output_bucket_name = sys.argv[1]


bucket_name = 'georgipanterov.data.patterns'
key_name = 'data_file.csv'
#aws_tools.create_bucket(bucket_name)
#key = aws_tools.store_private_data(bucket_name, key_name, 
#	inFile)
#print "Uploaded data to S3"

data_file_name = 's3_data_file.csv'
key2 = aws_tools.download_file(bucket_name, data_file_name, data_file_name )
print "Downloaded data from S3"

data = pd.read_csv(data_file_name, header=None, skiprows=1)
y = data.ix[:, 0]
train_data = data.ix[:, 1:]
N, Kall = np.shape(train_data)
train_data.columns = range(Kall)



# Fit Functions
def fit_func(chrom):
	return 1. * np.sum(chrom) / len(chrom)

def model_fit(chrom, y, train_data):
	X = train_data.ix[:, list(chrom)]
	adjR2 = models.gof_ols(y, X)
	n = np.sum(chrom)
	b = 0
	c = 5e-4
	penalty = b * n + c * (n ** 2)
	return  (1 + adjR2) * np.exp(-penalty)

fit_func2 = lambda chrom: model_fit(chrom, y, train_data)

pop_size =150
num_gens = 100

# length of chromosome
l = len(train_data.columns)

# Create population
pop = []
for i in range(pop_size):
	chrom = np.asarray(np.random.randint(0, 2, size=(l, )), dtype=bool)
	chrom = ga.make_chrom_valid(chrom)
	pop.append(tuple(chrom))
pop = tuple(pop)

all_best_fit = []

for i in range(num_gens):
	fit = ga.get_fitness(pop, fit_func2)
	pop = ga.offspring_pop(pop, fit, 
			mating_func = ga.mate_random_cross, mutate_prob=0.1)

	indx = np.argsort(fit)
	best = pop[indx[-1]]
	best_vars = np.where(np.array(best) == True)

	all_best_fit.append(fit[indx[-1]])
	if len(set(all_best_fit[-30:])) == 1 and len(all_best_fit) > 30:
		print "Keep getting same results so we break"
		break


	print "Fitness is: ", fit[indx[-1]], np.sum(best), \
					" ## ", best_vars[0][0:15]

##########
# Output #
##########
results_file_name = output_key_name
f = open(results_file_name, 'w')
#model_vars = [str(i) for i in best_vars]
#model_vars = '-'.join(model_vars)
#print model_vars
#f.write(model_vars)
X = train_data.ix[:, list(best)]
model = sm.OLS(y, X)
res = model.fit()
reg_output = res.summary()
f.write(str(reg_output))




f.close()


#output_bucket_name = 'georgipanterov.data.patterns.results'
aws_tools.create_bucket(output_bucket_name)
print "done with output bucket"
key = aws_tools.store_private_data(output_bucket_name, output_key_name, 
	results_file_name)


pop_file_name = 'pop_' + output_key_name
ga.write_pop_to_file(pop, pop_file_name)
pop_bucket = output_bucket_name + '.pop'

aws_tools.create_bucket(pop_bucket)
key = aws_tools.store_private_data(pop_bucket, pop_file_name, 
	pop_file_name)


