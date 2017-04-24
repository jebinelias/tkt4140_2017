import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time

def solvePoisson2D(p,b,bcType,bc= [0.,0.,0.,0.] ):
    """
    Solves Poisson problem for different boundary conditions
    b: is RHS of Poisson equation, should be mounted before calling
       this function
    """
    
    ny , nx = p.shape
    
    
    if bcType==1:
        # ADD CODE HERE
                
    
        
    return p

if __name__ == '__main__':  # this allows for the "main" part of the code to be
                            # executed only if the module is run as a program,
                            # while it allows to import it as a module
                            # in order to reuse functions defined here

    
    # viscosity
    nu = 0.05
    
    # space-time domain
    xMin = 0.
    xMax = 2.
    yMin = 0.
    yMax = 2.
    tIni = 0.
    tEnd = 1.
    # discretization of space-time domain
    nx = 100    # number of grid points for spatial mesh in x-direction
                 
    dx = (xMax-xMin)/(nx-1) # spatial mesh spacing
    
    ny = 100    # number of grid points for spatial mesh in y-direction
                 
    dy = (yMax-yMin)/(ny-1) # spatial mesh spacing
    
         
    x = np.linspace(xMin,xMax,nx) 
    y = np.linspace(yMin,yMax,ny)

    # solution arrays
    p = np.zeros((ny, nx)) ##create a 1xn vector of 1's
    b = np.zeros((ny, nx)) ##create a 1xn vector of 1's
    # mesh coordinates    
    X, Y = np.meshgrid(x, y)

    # initial condition
    # p=0 everywhere
    # boundary conditions
    bcType = 1 # 0: lid-driven cavity
    if bcType==1: # digital compendium
        bc = [1.,0.,0.,0.] # || N W S E
        p[-1,:] = 1.
        nyL = ny - 1 # because of Dirichlet boundary conditions at N 
        nxL = nx - 1 # because of Dirichlet boundary conditions at E
        n = (nxL)*(nyL) #number of unknowns
        b = np.zeros(n) #RHS
    # visualization
    plt.figure()
    CS = plt.contour(X, Y, p,20)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('2D Diffusion')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()
    plt.close()
    
    # solve Poisson equation
    # should mount b if not zero everywhere
    p = solvePoisson2D(p,b,bcType,bc)
    
    plt.figure()
    CS = plt.contour(X, Y, p,20)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('2D Poisson')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()
    
        
        
    print "DONE"
    
    
    
    
