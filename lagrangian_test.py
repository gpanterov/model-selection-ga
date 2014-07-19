from scipy.optimize import minimize
from scipy.optimize import brute
import numpy as np

def func(x):
	return 1.3*x[0]**2 + x[1]**2
x0 = np.ones(2)

# Unconstrainted optimization
sol = minimize(func, x0, method='SLSQP') 

# Constraint optimization
cons = ({'type':'eq',
		'fun' : lambda x: 5 - x[0] - x[1]},)
solc = minimize(func, x0, method='SLSQP', constraints=cons)

# Lagrangian
def lagrangian(xl):
	x = xl[1:]
	l = xl[0]
	return (1.3*x[0]**2 + x[1]**2-  l * (5 - x[0] - x[1]))

xl0 = np.ones(3)
soll = minimize(lagrangian, xl0, method='SLSQP')

# Brute force
rranges=(slice(-5, 5, 0.01), slice(-5, 5, 0.01))

#sol_brute = brute(func, rranges, full_output=True)

