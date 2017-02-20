# python_exercises/pexercise3/problem3a.py;ODEschemes.py @ git@lrhgit/tkt4140/src/src-ch2/ODEschemes.py;

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
N = 200
time = np.linspace(t0, tend, N)
h = time[1]-time[0]

U_0 = [um_py(t0), du_py(0)]

U_euler = euler(dUfunc, U_0, time)
u_euler = U_euler[:, 0]

U_heun = heun(dUfunc, U_0, time)
u_heun = U_heun[:, 0]

U_rk4 = rk4(dUfunc, U_0, time)
u_rk4 = U_rk4[:, 0]

U_heu_gen = heun_gen(dUfunc, U_0, time)
u_heu_gen = U_heu_gen[:, 0]
    
u_man = um_py(time)



plt.figure()

plt.plot(time, u_euler)
plt.plot(time, u_heun)
plt.plot(time, u_rk4)
plt.plot(time, u_heu_gen)
plt.plot(time, u_man, 'k--')
plt.legend(['euler', 'heun', 'rk4' ,'heun_gen', 'manufactured'], frameon=False,loc="best")
plt.show()

