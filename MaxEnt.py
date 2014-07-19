import numpy as np


def obj_me_moments(x, sim_params):
	"""
	Objective function with penalty for the ME problem
	x: 1d-array: the probabilities (p) for the model coefficients
	sim_params: dict with the parameters of the simulation
	"""
	#Entropy
	P = x.reshape((sim_params['K'], len(sim_params['psupport'])))
	Hp = -np.sum(np.sum(P * np.log(P), axis=1))
	# Proper probability constraint
	s = np.sum(P, axis=1)
	pbool = (s < 1 + 1e-10) * (s > 1 - 1e-10)
	penalty_prob = 1e5*(sim_params['K'] - np.sum(pbool))
	# Moment constraints
	params = np.sum(P * sim_params['psupport'], axis=1)
	e = sim_params['y'] - sim_params['fform'](params, sim_params['rel_vars'])
	e = e.reshape((sim_params['T'], 1))
	moments = np.sum(sim_params['rel_vars']*e, axis=0)
	penalty_moments = 1e5 * np.sum(moments ** 2)
	#penalty_moments = K - np.sum((moments < 0 + 1e-10) * (moments >0 - 1e-10))
	Obj = -Hp + penalty_prob + penalty_moments
	return Obj

def obj_gme_allsamples(x, sim_params):
	"""
	Objective function with penalty for the GME problem
	x: 1d-array: the probabilities (p) for the model coefficients
	sim_params: dict with the parameters of the simulation
	"""
	PW = x.reshape((sim_params['T'] + sim_params['K'], 
			len(sim_params['psupport'])))	
	P = PW[0:sim_params['K'], :]
	W = PW[sim_params['K']:, :]

	#Entropy
	Hp = -np.sum(np.sum(P * np.log(P), axis=1))
	# Proper probability constraint
	s = np.sum(P, axis=1)
	pbool = (s < 1 + 1e-10) * (s > 1 - 1e-10)
	penalty_prob = 1e5*(sim_params['K'] - np.sum(pbool))

	s2 = np.sum(W, axis=1)
	pbool2 =  (s2 < 1 + 1e-10) * (s2 > 1 - 1e-10)
	penalty_prob2 = 1e5*(sim_params['T'] - np.sum(pbool2))


	# Data constraints
	coefs = np.sum(P * sim_params['psupport'], axis=1)
	e = sim_params['y'] - sim_params['fform'](coefs, sim_params['rel_vars'])
	cons = e - np.sum(W * sim_params['esupport'], axis=1)
	pbool3 = (cons > -1e5) * (cons< 1e5)
	#penalty_data = 1e5 * (sim_params['T'] - np.sum(pbool3))
	#penalty_data = 1e5 * np.mean(e**2)
	penalty_data = 1e5 * np.mean(np.abs(e))
	Obj = -Hp + penalty_prob + penalty_prob2 + penalty_data
	return Obj



