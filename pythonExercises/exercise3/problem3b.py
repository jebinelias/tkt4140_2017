# python_exercises/pexercise3/problem3b.py;ODEschemes.py @ git@lrhgit/tkt4140/src/src-ch2/ODEschemes.py;

import numpy as np
import sympy as sp
import matplotlib.pylab as plt
from math import pi, sqrt
from ODEschemes import euler, heun, rk4

def dUfunc(U, tau):
    
    u0 = U[0]
    u1 = U[1]
    
    du0 = u1
    du1 = g_py(tau) - u0*u1

    return np.array([du0, du1])


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

def heun_gen(dUfunc, U_0, time):
    
    
    
    U = np.zeros((len(time), len(U_0)))
    U[0,:] = U_0
    
    for n, tau in enumerate(time[0:-1]):
        h = time[n + 1] - time[n]
        Un = U[n,:]
        
        k1 = dUfunc(Un, tau)
        k2 = dUfunc(Un + h*k1, tau + h)
        
        U[n + 1,:] = Un + h*(0.5*k1 + 0.5*k2)
    
    return U

# main program starts here:
t = sp.symbols('t')
um = sp.cos(sp.exp(t))

du = sp.diff(um, t)
ddu = sp.diff(du, t)

g = ddu + um*du

um_py = sp.lambdify(t, um, np)
du_py = sp.lambdify(t, du, np)
g_py = sp.lambdify(t, g, np)

t0, tend = 0, pi
N = 20
time = np.linspace(t0, tend, N)

U_0 = [um_py(t0), du_py(0)]

Ntds = 9
orderList = []

for i in range(Ntds):
     
    U = generic2ndOrder(dUfunc, U_0, time, a2=2./3) # a2 = 0.5 -> heun, a2 = 2/3 -> ralston, a2 = 1 -> midpoint
    u_num = U[:, 0]
    u_man = um_py(time)
 
    error = np.max(np.abs(u_num-u_man))
    if i>0:
        O = np.log(prev_error/error)/np.log(2)
        orderList.append(O)
    prev_error = error
    N *= 2
    time = np.linspace(t0, tend, N)
     
 
print orderList
 
plt.figure()
plt.plot(orderList)
plt.show()
 

