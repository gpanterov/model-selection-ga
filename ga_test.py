import numpy as np
from scipy.optimize import leastsq, minimize
import pandas as pd
import statsmodels.api as sm
import data_tools as tools
from sklearn import svm, linear_model
import time

reload(tools)



K = 10
N = 1000
f = tools.fform

operators = '+*+**+***'


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


