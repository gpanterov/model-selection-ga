import numpy as np
from scipy.optimize import leastsq, minimize
import pandas as pd
import statsmodels.api as sm
import data_tools as tools
reload(tools)


###############
# Create Data #
###############
rel_vars = np.random.normal(size=(n, k))
noise = np.random.normal( size=(n, ))
y = fform(params, rel_vars) + noise

data = np.column_stack((y, rel_vars))
df = pd.DataFrame(data)

new_data = np.random.normal(size=(n/3, k))
new_noise = np.random.normal(size=(n/3, ))
new_y = fform(params, new_data) + new_noise 


##################
# Estimate Model #
##################

# Inital values for optimization
x0 = np.ones(shape=(k,))
#x0 = np.random.normal(size=(k,))
sol = minimize(obj_func, x0, args=(y, fform, rel_vars), method='Nelder-Mead')

############
# Results #
############
print sol['message']
print "\n"

yfit = fform(sol['x'], new_data)
actual_mse = np.mean((new_y - fform(params, new_data)) ** 2)
estim_mse = np.mean((new_y - fform(sol['x'], new_data)) ** 2)
actual_pct_dev = np.mean(np.abs(new_y / fform(params, new_data)))
estim_pct_dev = np.mean(np.abs(new_y / fform(sol['x'], new_data)))

print "Out of sample TRUE MSE is: ", actual_mse
print "Out of sample ESTIMATED mse is: ", estim_mse
print "\n"

bias = np.mean(new_y - fform(sol['x'], new_data))
print "The bias of the estimator is: ", bias
print "\n"

res = np.column_stack((params, sol['x']))
print "Actual            Estimates"
print res

#############
# Benchmark #
#############

model = sm.OLS(y, rel_vars)
results = model.fit()
y_ols = results.predict(exog=new_data)
ols_mse = np.mean((new_y - y_ols)**2)
print "Out of sample OLS mse is: ", ols_mse





