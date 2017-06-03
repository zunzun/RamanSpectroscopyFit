import numpy as np
import pickle # for loading pickled test data
import deap # genetic algorithm library
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# peak function
def double_Lorentz(x, a, b, A0, w0, x_0, A1, w1, x_1):
    return a*x+b+(2*A0/np.pi)*(w0/(4*(x-x_0)**2 + w0**2))+(2*A1/np.pi)*(w1/(4*(x-x_1)**2 + w1**2))


# this will use DEAP to find initial parameter values
def generate_Initial_Parameters():
    return [0.07, 79, 233240, 13.24, 1591.5, 68090.96, 15.55, 1566.9] # original values



# load the pickled test data that wassaved from Raman spectroscopy
[xData, yData] = pickle.load(open('data.pkl', 'rb'))

# generate initial parameter values
initialParameters = generate_Initial_Parameters()

# curve fit the test data
fittedParameters, niepewnosci = curve_fit(double_Lorentz, xData, yData, initialParameters)

# create values for display of fitted peak function
a, b, A, w, x_0, A1, w1, x_01 = fittedParameters
y_fit = double_Lorentz(xData, a, b, A, w, x_0, A1, w1, x_01)

plt.plot(xData, yData) # plot the raw data
plt.plot(xData, y_fit) # plot the equation using the fitted parameters
plt.show()
