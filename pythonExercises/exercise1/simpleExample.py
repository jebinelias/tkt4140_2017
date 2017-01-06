""" In python there are a lot of different packages that have different functionality. 
    For instance the package numpy containt many functions etc. for handling linear algebra; 
    vectors, matrices, solving systems of linear equations.."""

import numpy as np

a = np.array([1,2, 3])
b = np.array([2, 4, 6])

c = a + b
print c
print a*a
print a**2 # a**2 -> a^2

# you can create functions:

def myConverter(USD, currency=8.48):
    
    """ Function that converts US dollars to NOK """
    NOK = USD*currency

    return NOK

# and run the function

NOK = myConverter(10)
NOK2 = myConverter(10, currency=8.0)

print "10 USD is NOK: ", NOK
