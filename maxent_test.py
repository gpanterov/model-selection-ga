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


def linear_fform2(params, rel_vars):
	dependent_var = params[0] * rel_vars[:, 0] + params[1] * rel_vars[:, 1] + \
					params[2] * rel_vars[:, 2]
	return dependent_var



def obj_func(x0, psupport, esupport):
	X0 = x0.reshape((T + K, len(psupport)))	
	P0 = X0[0:K, :]
	W0 = X0[K:, :]
	Hp = np.sum(np.sum(P0 * np.log(P0), axis=1))
	Hw = np.sum(np.sum(W0 * np.log(W0), axis=1))
	return Hp + Hw

def obj_func_me(x0, psupport):
	P = x0.reshape((K, len(psupport)))
	Hp = np.sum(np.sum(P0 * np.log(P0), axis=1))
	return Hp


def constraint_moments(x0, num_of_var, y, X, psupport, esupport):
	params = x0.reshape((T + K, len(psupport)))	
	P0 = params[0:K, :]
	W0 = params[K:, :]

	pz = P0 * psupport

	X0 = X[:,0].reshape((T,1))
	X1 = X[:,1].reshape((T,1))
	X2 = X[:,2].reshape((T,1))

	e = y - (np.sum(np.kron(X0, pz[0,:]), axis=1) + \
			 np.sum(np.kron(X1, pz[1,:]), axis=1) + \
		 	 np.sum(np.kron(X2, pz[2,:]), axis=1) + \
		 	 np.sum(W0 * esupport, axis=1) )
	return np.sum(X[:, num_of_var] * e)



def constraint_moments_me(x0, num_of_var, y, X, psupport, fform):
	P = x0.reshape((K, len(psupport)))	

	params = np.sum(P * psupport, axis=1)
	e = y - fform(params, X)
	return np.sum(X[:, num_of_var] * e)

def constraint_prob(x0):
	params = x0.reshape((T + K, len(psupport)))	
	s=np.sum(params, axis=1)
	pbool = (s < 1 + 1e-10) * (s > 1 - 1e-10)
	return T + K - np.sum(pbool)


def constraint_prob_me(x0):
	params = x0.reshape(( K, len(psupport)))	
	s=np.sum(params, axis=1)
	pbool = (s < 1 + 1e-10) * (s > 1 - 1e-10)
	return K - np.sum(pbool)


def normalize(nd_array, axis=0):
	min_vals = np.min(nd_array, axis=axis)
	max_vals = np.max(nd_array, axis=axis)
	normalized_array = (nd_array - min_vals) / (max_vals - min_vals)
	return normalized_array

def lagrangian_me(x0, y, X, psupport, fform):
	lambda_all = x0[-2 * K :]
	lambda_prob = lambda_all[0:K]
	lambda_mom = lambda_all[K:]
	x = x0[0: -2*K]

	P = x.reshape((K, len(psupport)))	
	obj = -obj_func_me(x, psupport)
	cons_p = 1 - np.sum(P, axis=1)
	cons_mom0 = constraint_moments_me(x, 0, y, X, psupport, fform)
	cons_mom1 = constraint_moments_me(x, 1, y, X, psupport, fform)
	cons_mom2 = constraint_moments_me(x, 2, y, X, psupport, fform)
	L = obj + np.sum(lambda_prob * cons_p) + \
		np.sum(lambda_mom * np.array([cons_mom0, cons_mom1, cons_mom2]))
	return -L
#########################################
# Initial Parameters for the simulation #
#########################################
#np.random.seed(12345)
T = 1000
K = 3
params = [1, 1.5, 2.5]
fform = nl_fform

###############
# Create Data #
###############
rel_vars = np.random.normal(size=(T, K))
noise = np.random.normal( size=(T, ))
y = fform(params, rel_vars) + noise


##################
# Estimate Model #
##################
esupport = np.array([-1e2, 0., 1e2])
psupport = np.array([-1e2, 0., 1e2])

P0 = np.ones(shape=[K, len(psupport)]) / 3.
W0 = np.ones(shape=(T, len(esupport))) / 3.

x0 = np.row_stack((P0,W0)).flatten()
x0me = P0.flatten()
x0lag = np.row_stack((P0, [1.,1.,1.], [1.,1.,1.])).flatten()


cons = ({'type': 'eq',
		 'fun': constraint_moments,
		 'args': (0, y, rel_vars, psupport, esupport)},
		{'type': 'eq',
		 'fun': constraint_moments,
		 'args': (1, y, rel_vars, psupport, esupport)},
		{'type': 'eq',
		 'fun': constraint_moments,
		 'args': (2, y, rel_vars, psupport, esupport)})


cons_me = ({'type': 'eq',
		 'fun': constraint_moments_me,
		 'args': (0, y, rel_vars, psupport, fform)},
		{'type': 'eq',
		 'fun': constraint_moments_me,
		 'args': (1, y, rel_vars, psupport, fform)},
		{'type': 'eq',
		 'fun': constraint_moments_me,
		 'args': (2, y, rel_vars, psupport, fform)},
		{'type': 'eq',
		 'fun': constraint_prob_me}, )


sol = minimize(obj_func_me, x0me, args=(psupport,),
				method='SLSQP', constraints=cons_me)

#sol = minimize(lagrangian_me, x0lag, args=(y, rel_vars, psupport, fform),
#				method='Powell')
###########
# Results #
###########
X = sol['x'].reshape((K, 3))
#X = sol['x'].reshape((K + 2, 3))
P = X[0:K, :]
print sol['message']

params_estim = np.sum(P * psupport, axis=1)
print "The true value of the parameters is: ", params
print "The ME parameter estimates are: ", params_estim
#############
# Benchmark #
#############

#model = sm.OLS(y, rel_vars)
#results = model.fit()
#y_ols = results.predict(exog=new_data)
#ols_mse = np.mean((new_y - y_ols)**2)
#print "Out of sample OLS mse is: ", ols_mse
