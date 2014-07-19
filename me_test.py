import numpy as np
import MaxEnt as me
from scipy.optimize import minimize
reload(me)


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



def linear_fform2(params, rel_vars):
	dependent_var = params[0] * rel_vars[:, 0] + params[1] * rel_vars[:, 1] + \
					params[2] * rel_vars[:, 2]
	return dependent_var


		

#########################################
# Initial Parameters for the simulation #
#########################################
#np.random.seed(12345)
sim_params = {'T': 1000,
		'K':3,
		'coef': [1, 1.5, 2.5],
		'fform':linear_fform2,
		'psupport' : np.array([-1e2, 0., 1e2])}


###############
# Create Data #
###############
sim_params['rel_vars'] = np.random.normal(size=(sim_params['T'], sim_params['K']))
noise = np.random.normal( size=(sim_params['T'], ))
sim_params['y'] = sim_params['fform'](sim_params['coef'], sim_params['rel_vars']) + noise

############
# Estimate #
############
P0 = np.ones(shape=[sim_params['K'], len(sim_params['psupport'])])\
			 / len(sim_params['psupport'])
x0 = P0.flatten()

sol = minimize(me.obj_me_moments, x0, args=(sim_params,),
				method='Nelder-Mead')
P = sol['x'].reshape((sim_params['K'], len(sim_params['psupport'])))
coef = np.sum(P * sim_params['psupport'], axis=1)

print "Estimated Coefficients are: ", coef
print "Actual coefficients are: ", sim_params['coef']


