# python_exercises/pexercise2/problem1.py

import numpy as np
import matplotlib.pylab as plt
from scipy.special import erf


def euler(func, y_0, time):
    """ Generic implementation of the euler scheme for solution of systems of ODEs (note that x may be interchanged by e.g. x):
            y0' = y1
            y1' = y2
                .
                .
            yN' = f(yN-1,..., y1, y0, t)
            
            method:
            y0^(n+1) = y0^(n) + h*y1
            y1^(n+1) = y1^(n) + h*y2
                .
                .
            yN^(n + 1) = yN^(n) + h*f(yN-1, .. y1, y0, t)

        Args:
            func(function): func(y, t) that returns y' at time t; [y1, y2,...,f(yn-1, .. y1, y0, t)] 
            y_0(array): initial conditions
            time(array): array containing the time to be solved for
        
        Returns:
            y(array): array/matrix containing solution for y0 -> yN for all timesteps"""

    y = np.zeros((np.size(time), np.size(y_0)))
    y[0,:] = y_0

    for i in range(len(time)-1):
        h = time[i+1] - time[i]
        y[i+1,:]=y[i,:] + np.asarray(func(y[i,:], time[i]))*h

    return y

def newton_func(y, x):
    """ function that returns the RHS of the Newton ODE:
    
        y' = 1 - 3x + y + x^2 + xy
        
        Args:
            y(array): array [y0, y1] at distance x
            x(float): current position
        
        Returns:
            dy(float): y' 
        """
        
    dy = 1 - 3*x + y + x**2 + x*y
    return dy

x = np.linspace(0, 1.5, 101) #allocate Space
y_0 = 0

Y = euler(newton_func, y_0, x)
Y_newton = x - x**2 + x**3/3 - x**4/6 + x**5/30 - x**6/45

e = np.exp(1)
pi = np.pi

Y_analytical = 3*np.sqrt(2*pi*e)*np.exp(x*(1 + x/2))*(erf(np.sqrt(2)*(1 + x)/2) - erf(np.sqrt(2)/2)) + 4*(1-np.exp(x*(1 + x/2))) - x

plt.figure()
plt.plot(x, Y_analytical)
plt.plot(x, Y)
plt.plot(x, Y_newton)
plt.legend(['analytical', 'euler', 'Newton'], loc='best', frameon=False)
plt.xlabel('x')
plt.ylabel('y')
#plt.savefig('problem1.png', transparent=True)
plt.show()
