# python_exercises/pexercise3/problem1_and_2.py

####    Sympy Exercise ####

from sympy import (
    symbols,   # define mathematical symbols for symbolic computing
    diff,      # differentiate expressions
    integrate, # integrate expressions
    Rational,  # define rational numbers
    lambdify,  # turn symbolic expressions into Python functions
    )
import numpy as np
import matplotlib.pyplot as plt
# change some default values to make plots more readable 
LNWDT=3; FNT=11
plt.rcParams['lines.linewidth'] = LNWDT; plt.rcParams['font.size'] = FNT
font = {'size' : 16}; plt.rc('font', **font)
#### Problem 1a ####
t, v0, g = symbols('t v0 g')
y = v0*t - Rational(1,2)*g*t**2
#### Problem 1b ####
dydt = diff(y, t)
dy2dt = diff(y, t, t)
y2 = integrate(dydt, t)
print "Velocity/dydt: ", dydt
print "Acceleration/dy2dt: ", dy2dt
print "y: ", y
print "y integrated from Velocity/dydt: ", y2
#### Problem 1c ####
from sympy import solve
roots = solve(y, t)
print "roots of y are: ", roots
print "y(root1): ", y.subs(t, roots[0])
print "y(root2): ", y.subs(t, roots[1])
#### Problem 1d ####
rootsFunc = lambdify([v0, g], roots) 
yFunc = lambdify([t, v0, g], y)
numericRoots = rootsFunc(5, 9.81)
t0 = numericRoots[0]
tend = numericRoots[1]
time = np.linspace(t0, tend, 101)

Y = np.zeros_like(time)
for i, tau in enumerate(time):
    Y[i] = yFunc(tau, 5, 9.81)
from sympy import latex
plt.figure()
plt.plot(time, Y)
plt.xlabel('time')
plt.ylabel('y')
plt.title('$ y = ' + latex(y) + '$')
#plt.savefig('../figs/sympy_exercise1d.png')
#### Problem 1e ####
yFuncArray = lambdify([t, v0, g], y, np) # adding np in the end will make yFuncArray able to take arrays as inputs
#### Problem 2a ####
from sympy import sin, cos, exp
f = exp(t)
print "taylor series of %s: " % f, f.series(t, 0, 5) # series is an attribute of any sympy expression;
#f.series(t, 0, 5) means calculating the 5 first terms of the taylor polynomial expansion arount t=0
#### Problem 2b ####
x = symbols('x')
f = sin(x)
space = np.linspace(0, 2*np.pi, 101)

plt.figure()
legendList = []
fFunc = lambdify(x, f, np) # create python function of the analytical solution 
plt.plot(space, fFunc(space), 'k--')
legendList.append("%s " % f)

for n in range(4, 13, 2):
    fTaylor = f.series(x, 0, n).removeO() # create sympy expression of the taylor series with n termes, and remove the residual term O.
    fTaylorfunc = lambdify(x, fTaylor, np) # create python function of the taylor series
    fapprox = fTaylorfunc(space) # calculate from 0 to 2pi
    
    plt.plot(space, fapprox)
    legendList.append('series, N = ' + str(n))

plt.xlabel('x')
plt.ylabel('f')
plt.xlim(space[0], space[-1])
plt.ylim(-2, 2)
#plt.title("f = %s, together with Taylor approximations of f" % f)
plt.legend(legendList, loc=3)
#plt.savefig('../figs/sympy_exercise2b.png')


plt.show()
        






