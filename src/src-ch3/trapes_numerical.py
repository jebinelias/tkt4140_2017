# coding: utf-8
# src-ch3/trapes_numerical.py;

import numpy as np
from matplotlib.pyplot import *
import scipy as sc
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import time


# change some default values to make plots more readable 
LNWDT=3; FNT=11
rcParams['lines.linewidth'] = LNWDT; rcParams['font.size'] = FNT
font = {'size' : 16}; rc('font', **font)

h = 0.1               # element size
L =1.0                  # length of domain
n = int(round(L/h))     # number of unknowns, assuming the RHS boundary value is known
x=np.arange(n+2)*h      # x includes min and max at boundaries were bc are imposed.

gamma=h/(2*(1+x[0:-2]))

#Create matrix for sparse solver
diagonals=np.zeros((3,n))
diagonals[0,:]= gamma-1                       #all elts in first row is set to 1
diagonals[1,:]= 2.0*(1+8.0*h*gamma)
diagonals[2,:]= -gamma-1.0 

#specific values for BC
diagonals[2,0]= -2
diagonals[1,0]= 2.0+h*(1+15.0*gamma[0]) 

A = sc.sparse.spdiags(diagonals, [-1,0,1], n, n,format='csc') #sparse matrix instance

#Crete rhs array
d=np.zeros(n)
d[n-1]=1+gamma[-1]


#Solve linear problems
tic=time.clock()
theta = sc.sparse.linalg.spsolve(A,d) #theta=sc.linalg.solve_triangular(A,d)
toc=time.clock()
print 'sparse solver time:',toc-tic


# Plot solutions
plot(x[:-2],theta)
legends=[] # empty list to append legends as plots are generated
legends.append(r'$\theta$')
ylabel(r'$\theta$')
xlabel('x')

show()
close()
print 'done'