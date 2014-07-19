import numpy as np
from scipy.optimize import leastsq, minimize
import pandas as pd
import statsmodels.api as sm
import data_tools as tools
from sklearn import svm, linear_model
import time

reload(tools)

def obj_func(x, data, operators, func, y):
	e = y - func(x, data, operators)
	mse = np.mean(e ** 2)
	return mse

K = 10
N = 1000
f = tools.fform

operators = '+*+**+***'
operators2 = '+*+**+***'
# operators2 differs slightly from operators. This is to mimic if we don't have
# exactly the right func form

Actual = []
NLLS = []
OLS = []
SVM = []
Lasso = []
WLS = []

start = time.time()
for i in range(100):
	if i%30 == 0:
		print i
	###############
	# Create Data #
	###############
	coefs = np.random.normal(size=(K, ))
	train_data = np.random.normal(size=(N, K))

	noise = np.random.normal( size=(N, ))
	y = f(coefs, train_data, operators) + noise

	# Create out-of-sample data
	new_data = np.random.normal(size=(N / 3, K))
	new_noise = np.random.normal(size=(N / 3, ))
	new_y = f(coefs, new_data, operators) + new_noise

	############
	# Estimate #
	############

	# NLLS
	x0 = np.ones(K)
	# !!!!!!!!!!! Make sure you change back operators2 to operators !!!!!!!
	sol = minimize(obj_func, x0, args=(train_data, operators2, f, y), \
						method='Nelder-Mead')

	# OLS
	model = sm.OLS(y, train_data)
	ols_res = model.fit()

	# SVM
	clf = svm.SVR()
	clf.fit(train_data, y)

	# Lasso
	clf_lasso = linear_model.Lasso()
	clf_lasso.fit(train_data, y)

	# WLS
	mod_wls = sm.WLS(y, train_data)
	res_wls = mod_wls.fit()

	################
	# Evaluate Fit #
	################
	actual_mse = np.mean((new_y - f(coefs, new_data, operators))**2)

	# !!!!!!!!!!! Make sure you change back operators2 to operators !!!!!!!
	nlls_mse = np.mean((new_y - f(sol['x'], new_data, operators2))**2)
	ols_mse = np.mean((new_y - ols_res.predict(new_data))**2)
	svm_mse = np.mean((new_y - clf.predict(new_data))**2)
	lasso_mse = np.mean((new_y - clf_lasso.predict(new_data))**2)
	wls_mse = np.mean((new_y - res_wls.predict(new_data))**2)


	Actual.append(actual_mse)
	NLLS.append(nlls_mse)
	OLS.append(ols_mse)
	SVM.append(svm_mse)
	Lasso.append(lasso_mse)
	WLS.append(wls_mse)

	#print "Actual MSE: ", actual_mse
	#print "NLLS MSE: ", nlls_mse
	#print "OLS MSE: ", ols_mse
	#print "SVM MSE: ", svm_mse
	#print "\n"
	#print "Actual: ", coefs
	#print "Estimated: ", sol['x']
R = np.column_stack((Actual, NLLS, OLS, SVM, Lasso, WLS))
best_nlls = np.sum(np.min(R[:,1:], axis=1) == R[:,1])
best_ols =  np.sum(np.min(R[:,1:], axis=1) == R[:,2])
best_svm = np.sum(np.min(R[:,1:], axis=1) == R[:,3])
best_lasso = np.sum(np.min(R[:,1:], axis=1) == R[:,4])
best_wls = np.sum(np.min(R[:,1:], axis=1) == R[:,5])

print "NLLS was the best: ", best_nlls, " times"
print "OLS was the best: ", best_ols, " times"
print "SVM was the best: ", best_svm, " times"
print "Lasso was the best: ", best_lasso, " times"
print "WLS was the best: ", best_wls, " times"
print "\n"
print "The entire simulation took", time.time() - start, " seconds"
