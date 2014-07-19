import numpy as np
from scipy.optimize import leastsq, minimize
import pandas as pd
import statsmodels.api as sm

def nl_fform(params, rel_vars):
	dependent_var =  params[0] * rel_vars[:, 0] + \
					np.exp( params[1] * rel_vars[:, 1]) * \
					(params[2] * rel_vars[:, 2]) 
	return dependent_var

def nl_fform2(params, rel_vars):
	dependent_var =  params[0] * rel_vars[:, 0] * \
					np.exp( params[1] * rel_vars[:, 1]) * \
					(params[2] * rel_vars[:, 2]) 
	return dependent_var


def nl_fform3(params, rel_vars):
	dependent_var =  params[0] * rel_vars[:, 0] * \
					np.exp( params[1] * rel_vars[:, 1]) * \
					(params[2] * rel_vars[:, 2]) 
	return dependent_var


def linear_fform(params, rel_vars):
	dependent_var = params[0] * rel_vars[:, 0] + params[1] * rel_vars[:, 1] + \
					params[2] * rel_vars[:, 2] + params[3] * rel_vars[:, 3] + \
					params[4] * rel_vars[:, 4]
	return dependent_var

def linear_fform2(params, rel_vars):
	dependent_var = params[0] * rel_vars[:, 0] + params[1] * rel_vars[:, 1] + \
					params[2] * rel_vars[:, 2]
	return dependent_var

def sine_fform(params, rel_vars):
	dependent_var = params[0] * rel_vars[:, 0] * np.sin(params[1] * \
					rel_vars[:, 1]) * \
					np.exp(params[2] * rel_vars[:, 2])
	return dependent_var


def obj_func(params, y, fform, data):
	e = y - fform(params, data)
	mse = np.mean(e ** 2)
	return mse
def normalize(nd_array, axis=0):
	min_vals = np.min(nd_array, axis=axis)
	max_vals = np.max(nd_array, axis=axis)
	normalized_array = (nd_array - min_vals) / (max_vals - min_vals)
	return normalized_array

#########################################
# Initial Parameters for the simulation #
#########################################
#np.random.seed(12345)
n = 2000
k = 3
params = [1, 1.5, 2.5]


fform = nl_fform

###############
# Create Data #
###############
rel_vars = np.random.normal(size=(n, k))
noise = np.random.normal( size=(n, ))
y = fform(params, rel_vars) + noise

data = np.column_stack((y, rel_vars))
df = pd.DataFrame(data)

new_data = np.random.normal(size=(n/3, k))
new_noise = np.random.normal(size=(n/3, ))
new_y = fform(params, new_data) + new_noise 


##################
# Estimate Model #
##################

# Inital values for optimization
x0 = np.ones(shape=(k,))
#x0 = np.random.normal(size=(k,))
sol = minimize(obj_func, x0, args=(y, fform, rel_vars), method='Nelder-Mead')

############
# Results #
############
print sol['message']
print "\n"

yfit = fform(sol['x'], new_data)
actual_mse = np.mean((new_y - fform(params, new_data)) ** 2)
estim_mse = np.mean((new_y - fform(sol['x'], new_data)) ** 2)
actual_pct_dev = np.mean(np.abs(new_y / fform(params, new_data)))
estim_pct_dev = np.mean(np.abs(new_y / fform(sol['x'], new_data)))

print "Out of sample TRUE MSE is: ", actual_mse
print "Out of sample ESTIMATED mse is: ", estim_mse
print "\n"

bias = np.mean(new_y - fform(sol['x'], new_data))
print "The bias of the estimator is: ", bias
print "\n"

res = np.column_stack((params, sol['x']))
print "Actual            Estimates"
print res

#############
# Benchmark #
#############

model = sm.OLS(y, rel_vars)
results = model.fit()
y_ols = results.predict(exog=new_data)
ols_mse = np.mean((new_y - y_ols)**2)
print "Out of sample OLS mse is: ", ols_mse





