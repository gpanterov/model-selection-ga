import pyOpt
import numpy as np
from numpy import log
#!/usr/bin/env python
'''
Solves Schittkowski's TP37 Problem.

    min 	-x1*x2*x3
    s.t.:	x1 + 2.*x2 + 2.*x3 - 72 <= 0
            - x1 - 2.*x2 - 2.*x3 <= 0
            0 <= xi <= 42,  i = 1,2,3
    
    f* = -3456 , x* = [24, 12, 12]
'''

# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, time
import pdb

# =============================================================================
# Extension modules
# =============================================================================
#from pyOpt import *
from pyOpt import Optimization
from pyOpt import PSQP
from pyOpt import SLSQP
from pyOpt import CONMIN
from pyOpt import COBYLA
from pyOpt import SOLVOPT
from pyOpt import KSOPT
from pyOpt import NSGA2


# =============================================================================
# 
# =============================================================================
def objfunc(x):
    
    f = -x[0]*x[1]*x[2]
    g = [0.0]*2
    g[0] = x[0] + 2.*x[1] + 2.*x[2] - 72.0
    g[1] = -x[0] - 2.*x[1] - 2.*x[2]
    
    fail = 0
    return f,g, fail
    

# =============================================================================
# 
# =============================================================================

# Instantiate Optimization Problem 
opt_prob = Optimization('TP37 Constraint Problem',objfunc)
opt_prob.addVar('x1','c',lower=0.0,upper=42.0,value=10.0)
opt_prob.addVar('x2','c',lower=0.0,upper=42.0,value=10.0)
opt_prob.addVar('x3','c',lower=0.0,upper=42.0,value=10.0)
opt_prob.addObj('f')
opt_prob.addCon('g1','i')
opt_prob.addCon('g2','i')
print opt_prob

# Instantiate Optimizer (PSQP) & Solve Problem
psqp = PSQP()
psqp.setOption('IPRINT',0)
psqp(opt_prob,sens_type='FD')
print opt_prob.solution(0)

def objfunc2(x):
	return - x[0]*x[1]*x[2]
cons = ({'type': 'ineq',
		'fun': lambda x: -(x[0] + 2.*x[1] + 2.*x[2] - 72.0)},
		{'type' : 'ineq',
		'fun': lambda x: -(-x[0] - 2.*x[1] - 2.*x[2])},
		{'type':'ineq',
		'fun': lambda x: x[0]},
		{'type':'ineq',
		'fun': lambda x: x[1]},
		{'type':'ineq',
		'fun': lambda x: x[2]},
)
#from scipy.optimize import minimize
#x0= np.ones(3)
#sol=minimize(objfunc2, x0, method='SLSQP', constraints=cons)
def nl_fform2(params, rel_vars):
	dependent_var =  params[0] * rel_vars[:, 0] * \
					np.exp( params[1] * rel_vars[:, 1]) * \
					(params[2] * rel_vars[:, 2]) 
	return dependent_var

def linear_fform2(params, rel_vars):
	dependent_var = params[0] * rel_vars[:, 0] + params[1] * rel_vars[:, 1] + \
					params[2] * rel_vars[:, 2]
	return dependent_var
def nl_fform(params, rel_vars):
	dependent_var =  params[0] * rel_vars[:, 0] + \
					np.exp( params[1] * rel_vars[:, 1]) * \
					(params[2] * rel_vars[:, 2]) 
	return dependent_var


#########################################
# Initial Parameters for the simulation #
#########################################
#np.random.seed(12345)
T = 1000
K = 3
params = [1, 1.5, 2.5]
fform =nl_fform 

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
X = rel_vars
x0 = P0.flatten()
def objfunc_ME(x):
	Hp = x[0]*log(x[0]) + x[1]*log(x[1]) + x[2]* log(x[2]) + \
		x[3]*log(x[3]) + x[4]*log(x[4]) + x[2]* log(x[5]) + \
		x[6]*log(x[6]) + x[7]*log(x[7]) + x[8]* log(x[8])  
	f = Hp
	g=[0.]*9
	g[0] = np.sum(np.array([x[0],x[1],x[2]])) - 1
	g[1] = np.sum(np.array([x[3],x[4],x[5]]) ) - 1
	g[2] = np.sum(np.array([x[6],x[7],x[8]])) - 1
	b1=np.sum(np.array([x[0],x[1],x[2]]) * psupport)
	b2=np.sum(np.array([x[3],x[4],x[5]]) * psupport)
	b3=np.sum(np.array([x[6],x[7],x[8]]) * psupport)
	params = np.array([b1,b2,b3])
	e = y - fform(params, X)
	g[3] = np.sum(X[:,0] * e)
	g[4] =  np.sum(X[:,1] * e)
	g[5] =  np.sum(X[:,2] * e)
	g[6] =1- np.sum(np.array([x[0],x[1],x[2]])) 
	g[7] =1- np.sum(np.array([x[3],x[4],x[5]]) ) 
	g[8] =1- np.sum(np.array([x[6],x[7],x[8]])) 

	fail = 0
	return f, g, fail

# Instantiate Optimization Problem 
opt_prob = Optimization('ME Constraint Problem',objfunc_ME)
opt_prob.addVar('x1','c',lower=0.0,upper=.999,value=0.333)
opt_prob.addVar('x2','c',lower=0.0,upper=.999,value=0.333)
opt_prob.addVar('x3','c',lower=0.0,upper=.999,value=0.333)
opt_prob.addVar('x4','c',lower=0.0,upper=.999,value=0.333)
opt_prob.addVar('x5','c',lower=0.0,upper=.999,value=0.333)
opt_prob.addVar('x6','c',lower=0.0,upper=.999,value=0.333)
opt_prob.addVar('x7','c',lower=0.0,upper=.999,value=0.333)
opt_prob.addVar('x8','c',lower=0.0,upper=.999,value=0.333)
opt_prob.addVar('x9','c',lower=0.0,upper=.999,value=0.333)



opt_prob.addObj('f')
opt_prob.addCon('g1','i')
opt_prob.addCon('g2','i')
opt_prob.addCon('g3','i')
opt_prob.addCon('g4','i')
opt_prob.addCon('g5','i')
opt_prob.addCon('g6','i')
opt_prob.addCon('g7','i')
opt_prob.addCon('g8','i')
opt_prob.addCon('g9','i')


print opt_prob

# Instantiate Optimizer (SLSQP) & Solve Problem
slsqp = SLSQP()
slsqp.setOption('IPRINT',-1)
slsqp(opt_prob,sens_type='FD')
print opt_prob.solution(0)

