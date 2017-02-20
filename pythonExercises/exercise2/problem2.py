# python_exercises/pexercise2/problem2.py

import numpy as np
import matplotlib.pylab as plt
#from myanimation import animatePendulum

def f(y, t):
    """Two by two system for mathematical pendulum"""
    y_out = np.zeros_like(y)
    y_out[:] = [y[1], -np.sin(y[0])]
    return y_out

# define Heun solver
def heun(func, z0, time):
    """The Heun scheme for solution of systems of ODEs.
    z0 is a vector for the initial conditions,
    the right hand side of the system is represented by func which returns
    a vector with the same size as z0 ."""

    z = np.zeros((np.size(time), np.size(z0)))
    z[0,:] = z0
    zp = np.zeros_like(z0)

    for i, t in enumerate(time[0:-1]):
        dt = time[i+1] - time[i]
        zp = z[i,:] + np.asarray(func(z[i,:],t))*dt   # Predictor step
        z[i+1,:] = z[i,:] + (np.asarray(func(z[i,:],t)) + np.asarray(func(zp,t+dt)))*dt/2.0 # Corrector step

    return z
        

# main program starts here

theta0deg = 85.
theta0dot = 0
theta0 = theta0deg*np.pi/180.0

Npts = 50
time = np.linspace(0.0, 15.0, Npts)

z0 = np.array([theta0, theta0dot])

# calculating response using Heun

zHeunNonlin = heun(f, z0, time)

# calculating energy function (zero, if exact solution)

E_heunNonlin = 0.5*(zHeunNonlin[:,1]**2 - theta0dot**2) + np.cos(theta0) - np.cos(zHeunNonlin[:,0])


# plot the amplitude
plt.figure(1)
plt.hold('on')

plt.plot(time, zHeunNonlin[:,0])
plt.ylabel('Amplitude [rad]')
plt.xlabel('Dimensionless time [-]')
#plt.savefig('amplitude.png', transparent=True)

# plot the energy function
plt.figure(2)
plt.hold('on')
plt.plot(time, E_heunNonlin)
plt.ylabel('Energy function [-]')
plt.xlabel('Dimensionless time [-]')
#plt.savefig('energy_function.png', transparent=True)

plt.show()
#animatePendulum(zHeunNonlin[:,0], E_heunNonlin, time, l=1)
