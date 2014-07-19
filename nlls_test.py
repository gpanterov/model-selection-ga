import numpy as np
from scipy.optimize import leastsq, minimize
import pandas as pd
import statsmodels.api as sm
import data_tools as tools
reload(tools)

def obj_func(params, y, fform, data):
	e = y - fform(params, data)
	mse = np.mean(e ** 2)
	return mse

K = 5
N = 1000
###############
# Create Data #
###############

operators = '++*+'


coefs = np.random.normal(size=(K, ))
train_data = np.random.normal(size=(N, K))

noise = np.random.normal( size=(N, ))
y = tools.fform(coefs, train_data, operators) + noise



