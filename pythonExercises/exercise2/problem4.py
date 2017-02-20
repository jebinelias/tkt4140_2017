# python_exercises/pexercise2/problem3.py

import numpy as np
import matplotlib.pylab as plt

def pendulum_lin(y, t):
    """ function that returns the RHS of the mathematcal pendulum (linearized version) ODE:    
        Reduction of higher order ODE:
        theta = y0
        theta' = y1
        theta'' = - theta = -y0
    
        y0' = y1
        y1' = -y0
        
        Args:
            y(array): array [y0, y1] at time t
            t(float): current time
        
        Returns:
            dy(array): [y0', y1'] = [y1, -y0]
        """
        
    dy = np.zeros_like(y)
    dy[:] = [y[1], -y[0]]
    return dy


def pendulum_nonlin(y, t):
    """ function that returns the RHS of the mathematcal pendulum (original version) ODE:    
        Reduction of higher order ODE:
        theta = y0
        theta' = y1
        theta'' = - sin(theta) = -sin(y0)
    
        y0' = y1
        y1' = -y0
        
        Args:
            y(array): array [y0, y1] at time t
            t(float): current time
        
        Returns:
            dy(array): [y0', y1'] = [y1, -sin(y0)]
        """
    y_out = np.zeros_like(y)
    y_out[:] = [y[1], -np.sin(y[0])]
    return y_out


# define Euler solver
def euler(func, y_0, time):
    """ Generic implementation of the euler scheme for solution of systems of ODEs:
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
        dt = time[i+1] - time[i]
        y[i+1,:]=y[i,:] + np.asarray(func(y[i,:], time[i]))*dt

    return y


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

def generic2ndOrder(func, z0, time, a2=0.5):
    """a2 = 0.5 -> heun
       a2 = 2/3 -> ralston
       a2 = 1 -> midpoint"""
       
    a1 = 1. - a2
    p1 = 0.5/a2
    p2 = 0.5/a2
    
    z = np.zeros((np.size(time), np.size(z0)))
    z[0,:] = z0

    for i, t in enumerate(time[0:-1]):
        h = time[i+1] - time[i]
        k1 = np.asarray(func(z[i,:], t))
        k2 = np.asarray(func(z[i,:] + p2*k1*h, t + p1*h))
        z[i+1,:] = z[i,:] + (a1*k1 + a2*k2)*h

    return z
        

# main program starts here

theta0deg = 85
theta0dot = 0
theta0 = theta0deg*np.pi/180.0

Npts = 500
time = np.linspace(0.0, 15.0, Npts)

z0 = np.array([theta0, theta0dot])

zEulerLin = euler(pendulum_lin, z0, time)
zHeunLin = heun(pendulum_lin, z0, time)

# calculating response using generic 2nd Order scheme
zHeunGenLin = generic2ndOrder(pendulum_lin, z0, time)
zRalstonLin = generic2ndOrder(pendulum_lin, z0, time, a2=2./3) 
zMidpointLin = generic2ndOrder(pendulum_lin, z0, time, a2=1)

# plot the amplitude
plt.figure(1)
plt.hold('on')

plt.plot(time, zEulerLin[:,0])
plt.plot(time, zHeunLin[:,0])
plt.plot(time, zHeunGenLin[:,0])
plt.plot(time, zRalstonLin[:,0])
plt.plot(time, zMidpointLin[:,0])
plt.legend(['EulerLin', 'HeunLin', 'HeunGenLin', 'RalstonLin', 'MidpointLin'], frameon=False, loc='best')
plt.ylabel('Amplitude [rad]')
plt.xlabel('Dimensionless time [-]')
#plt.savefig('problem3.png', transparent=True)

plt.show()