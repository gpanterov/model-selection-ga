import numpy as np
from scipy.optimize import leastsq, minimize
import statsmodels.api as sm
from sklearn import svm, linear_model


def gof_ols(y, X):
	N, K = np.shape(X)
	cutoff = 2 * N / 3
	# Model training data
	X0 = X.ix[0: cutoff, :]
	y0 = y.ix[0: cutoff]

	# Model evaluation data
	X1 = X.ix[cutoff:, :]
	y1 = y.ix[cutoff:]

	# Model estimation
	model = sm.OLS(y0, X0)
	results = model.fit()

	# Evaluate fit
	yhat = results.predict(X1)
	SSresid = np.sum((y1 - yhat) ** 2)
	SStot = np.sum((y1 - np.mean(y1))**2)
	R2 = 1 -  SSresid / SStot
	adjR2 = R2 - (1 - R2) * (K / (N - K - 1))
	
	#mse = np.mean((y1 - yhat)**2)	

	return adjR2


