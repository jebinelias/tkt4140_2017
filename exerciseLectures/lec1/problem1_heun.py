'''
Created on Feb 5, 2019

@author: fredrik
'''
import numpy as np
import matplotlib.pyplot as plt
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
    
    z_p = z[n] + dt*v[n]
    v_p = v[n] + dt*(-(2*g*z[n]/L + f*v[n]*abs(v[n])/(2*D)))
    
    z[n + 1] = z[n] + dt*(v[n] + v_p)/2
    v[n + 1] = v[n] + dt*((-(2*g*z[n]/L + f*v[n]*abs(v[n])/(2*D))) -(2*g*z_p/L + f*v_p*abs(v_p)/(2*D)))/2
    t[n + 1] = t[n] + dt
    
    print "z({0}) = {1}".format(t[n + 1], z[n + 1])
    
plt.figure()
plt.plot(t, z)
plt.xlabel("t")
plt.ylabel("z")
plt.show()