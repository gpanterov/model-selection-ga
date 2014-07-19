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
obj_func = me.obj_gme_allsamples
sim_params = {'T': 100,
		'K':3,
		'coef': [1, 1.5, 2.5],
		'fform':linear_fform2,
		'psupport' : np.array([-1e2, 0., 1e2]),
		'esupport' :  np.array([-1e3, 0., 1e3])}


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
W0 = np.ones(shape=[sim_params['T'], len(sim_params['esupport'])])\
			 / len(sim_params['esupport'])

x0 = np.row_stack((P0, W0)).flatten()

sol = minimize(obj_func, x0, args=(sim_params,),
				method='Nelder-Mead')
###########
# Results #
###########
PW = sol['x'].reshape((sim_params['K'] + sim_params['T']
						, len(sim_params['psupport'])))
P = PW[0:sim_params['K'], :]
W = PW[sim_params['K']:, :]

coef = np.sum(P * sim_params['psupport'], axis=1)

print "Estimated Coefficients are: ", coef
print "Actual coefficients are: ", sim_params['coef']


