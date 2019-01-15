'''
Created on Jan 14, 2019

@author: fredrik
'''
import matplotlib.pylab as plt
import numpy as np


b = np.array([12,5,2])
A = np.array([[-3,2,-4],[0,1,2],[2,4,5]])

A_inv = np.linalg.inv(A)

print "b + b", b + b
print "np.dot(b, b)", np.dot(b, b)
print "np.dot(A, A)", np.dot(A, A)
print "np.transpose(A)", np.transpose(A)
print "np.dot(A_inv, b)", np.dot(A_inv, b)
print "np.linalg.solve(A, b)", np.dot(A_inv, b)

x = np.linspace(0, 4*np.pi, 101)

plt.figure()
plt.plot(x, np.sin(x))
plt.plot(x, np.exp(-0.1*x)*np.sin(x))
plt.show()
