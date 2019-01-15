'''
Created on Jan 14, 2019

@author: fredrik
'''
from math import pi, sqrt
import matplotlib.pylab as plt

b = 0.1 # width [m]
h = 0.2 # height [m]
r = 0.15 # 0.0797884560803
L = 1. # length[]
dx = 0.1 # step length
P = 1*1e6
E_steel = 200*1e9
E_alu = 69*1e9

I_rectangular = b*h**3/12.

I_circular = pi*r**4/4.

y_tip = -L**3*P/(3*E_steel*I_rectangular)
print "tip deflection Steel, rectangular [mm]: ", y_tip*1e3

y_tip = -L**3*P/(3*E_steel*I_circular)
print "tip deflection Steel, circular [mm]: ", y_tip*1e3

y_tip = -L**3*P/(3*E_alu*I_rectangular)
print "\ntip deflection aluminium, rectangular [mm]:", y_tip*1e3

y_tip = -L**3*P/(3*E_alu*I_circular)
print "tip deflection aluminium, circular [mm]: ", y_tip*1e3

print "\narea rectangular: ", b*h, "area circular: ", pi*r**2

print  "r = ", sqrt(b*h/pi)

N = int(L/dx) + 1
X = []

for n in range(N):
    x = dx*n
    X.append(x)
    #print n, x
    
Y_steel = []
Y_alu = []

for x in X:
    y_steel = P*(-x**3 + 3*L**2*x - 2*L**3)/(6*E_steel*I_rectangular)*1e3
    Y_steel.append(y_steel)
    
    y_alu = P*(-x**3 + 3*L**2*x - 2*L**3)/(6*E_alu*I_rectangular)*1e3
    Y_alu.append(y_alu)
    
plt.figure()
plt.plot(X, Y_steel)
plt.plot(X, Y_alu)
plt.legend(["steel", "alu"])
plt.xlabel("x")
plt.ylabel("y [mm]")
plt.show()