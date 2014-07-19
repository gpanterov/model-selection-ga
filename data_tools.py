import numpy as np
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
		val = dict_op[op](val, bX[:, i+1])
	return val
		
