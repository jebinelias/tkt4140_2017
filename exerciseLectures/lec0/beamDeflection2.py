'''
Created on Jan 14, 2019

@author: fredrik
'''
import numpy as np
from math import pi
import matplotlib.pylab as plt

def calcDeflectionSteel(x):
    
    b = 0.1
    h = 0.2
    L = 1.
    I = b*h**3/12
    E = 200*1e9
    
    P = 1e6
    
    y = -P*(-x**3 + 3*L**2*x - 2*L**3)/(6*E*I)
    
    return y

def calcDeflectionAlu(x):
    
    b = 0.1
    h = 0.2
    L = 1.
    I = b*h**3/12
    E = 69*1e9
    
    P = 1e6
    
    y = -P*(-x**3 + 3*L**2*x - 2*L**3)/(6*E*I)
    
    return y

def calcDeflection(x, materialType, crossSectionType, b=0.1, h=0.2, r=0.15, L=1, P=1e6):
    
    if crossSectionType == "rectangular":
        I = b*h**3/12
    elif crossSectionType == "circular":
        I = pi*r**4/4.
    else:
        print "Warning, crossSectionType type: ", crossSectionType, " is not available"
        exit(-1)
        
    if materialType == "steel":
        E = 200*1e9
    elif materialType == "alu":
        E = 69*1e9
    else:
        print "Warning, material type: ", materialType, " is not available"
        exit(-1)
    
    y = P*(-x**3 + 3*L**2*x - 2*L**3)/(6*E*I)
    
    return y

L = 1.
dx = 0.1
materialType = "alu"
crossSectionType = "rectangular"

N = int(L/dx) + 1

X = []
Y = []

for n in range(N):
    x = n*dx
    y = calcDeflection(x, materialType, crossSectionType)*1e3
    X.append(x)
    Y.append(y)

X_numpy = np.linspace(0, L, N)
Y_numpy = calcDeflection(X_numpy, materialType, crossSectionType)*1e3

plotResultsSingle = False
plotResultsMultiple = True

if plotResultsSingle:
    plt.figure()
    plt.plot(X, Y, "r")
    plt.plot(X_numpy, Y_numpy, "b--")
    plt.xlabel("x [m]")
    plt.ylabel("y [mm]")
    plt.title("material: " + materialType + ", cross-section: " + crossSectionType)
    plt.show()

if plotResultsMultiple:
    
    materialList = ["steel", "alu"]
    crossSectionList = ["rectangular", "circular"]
    plt.figure()
    legendList = []
    
    for materialType in materialList:
        for crossSectionType in crossSectionList:
            Y = calcDeflection(X_numpy, materialType, crossSectionType)*1e3
            plt.plot(X, Y)
            legendList.append(materialType + ", " + crossSectionType)

    plt.xlabel("x [m]")
    plt.ylabel("y [mm]")
    plt.legend(legendList)
    plt.show()
            
