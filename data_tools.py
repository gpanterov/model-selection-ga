import numpy as np
import pandas as pd
dict_op ={'*':np.prod,
			'+':np.sum}
def fform(coef, data, operators):
	"""
	Calculates the value of the target variable
	for the functional form described in the input parameters
	coef: Kx1 array, the coefficients multiplying the independent vars
	data: NxK array, the dependent variables
	operators: string with the operations that descibe the relationship between
				the variables

	returns y: Tx1 array of the target variable
	"""
	bX = data * coef
	val = bX[:,0]
	for i, op in enumerate(operators):
		if op == '*':
			val *= bX[:, i+1]
		if op == '+':
			val += bX[:, i+1]
		if op == '^':
			val **=bX[:, i+1]

	return val	

def create_data(fname, N, K, K2, f, operators, coefs):	
	train_data = np.random.normal(size=(N, K))

	noise = np.random.normal( size=(N, ))
	y = f(coefs, train_data, operators) + noise

	# Crate other (non-relevant) data
	other_data = np.random.normal( size=(N,K2))
	all_data = np.column_stack((y, train_data, other_data))
	pd.DataFrame(all_data).to_csv(fname, index=False, 
									header=False)

	train_data = pd.DataFrame(np.column_stack((train_data, other_data)))
	y = pd.Series(y)
	return y, train_data
