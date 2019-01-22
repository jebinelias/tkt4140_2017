'''
Created on Jan 22, 2019

@author: fredrik
'''
import numpy as np
import matplotlib.pylab as plt
D = 0.5 # diameter
f = 0.05 # friction
L = 392 # length
g = 9.8 # gravitational constant

dt = 1 # time-step
T = 3. # final time

z_0 = -6 # initial condtion for z
v_0 = 0

N = int(T/dt)

z = np.zeros(N + 1)
v = np.zeros(N + 1)
t = np.zeros(N + 1)

z[0] = z_0
v[0] = v_0
for n in range(N):
    
    z[n + 1] = z[n] + dt*v[n]
    v[n + 1] = v[n] + dt*(-(2*g*z[n]/L + f*v[n]*abs(v[n])/(2*D)))
    
    t[n + 1] = t[n] + dt
    print "z({0}) = {1}".format(t[n + 1], z[n + 1])
    print "v({0}) = {1}".format(t[n + 1], v[n + 1])

plt.figure()
plt.plot(t, z)
plt.xlabel("t")
plt.ylabel("z")
plt.show()
    