'''
Created on Feb 5, 2019

@author: fredrik
'''
import numpy as np
import matplotlib.pyplot as plt

def fFunc(t, z, v):
    
    dzdt = v
    dvdt = -(2*g*z/L + f*v*abs(v)/(2*D))
    
    return dzdt, dvdt

D = 0.5 # diameter [m]
f = 0.05 # friction coefficient
L = 392 # length [m]
g = 9.8 # gravitational constant

dt = 1
T = 3

N = int(T/dt)

z_0 = -6
v_0 = 0

z = np.zeros(N + 1)
v = np.zeros(N + 1)
t = np.zeros(N + 1)

z[0] = z_0
v[0] = v_0
for n in range(N):
    
    dzdt, dvdt = fFunc(t, z[n], v[n])
    z_p = z[n] + dt*dzdt
    v_p = v[n] + dt*dvdt
    
    dzdt_p, dvdt_p = fFunc(t, z_p, v_p)
    
    z[n + 1] = z[n] + dt*(dzdt + dzdt_p)/2
    v[n + 1] = v[n] + dt*(dvdt + dvdt_p)/2
    t[n + 1] = t[n] + dt
    
    print "z({0}) = {1}".format(t[n + 1], z[n + 1])
    
plt.figure()
plt.plot(t, z)
plt.xlabel("t")
plt.ylabel("z")
plt.show()