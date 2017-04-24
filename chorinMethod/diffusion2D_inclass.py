import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time

def mountDiffusion1D(N,dt,theta,D):
    """
    Mount matrix A for solving the 1D heat equation
    using theta-scheme and Dirichlet boudanry conditions
    at both sides of the 1D domain
    N: is the number of grid points 
    D: nu*dt/dx/dx
    A: linear system matrix for theta-scheme discretization of
       heat quation
    """
    
    # ADD CODE HERE:
    # - use sparse matrix format, check Digital compendium example
    # - do not forget boundary conditions impact on A 

    return A



def evolveDiffusion1D(u,theta,D,A):
    """
    Evolve the 1D heat equation
    using theta scheme and Dirichlet boudanry conditions
    at both sides of the 1D domain
    D: nu*dt/dx/dx
    A: linear system matrix for theta-scheme discretization of
       heat quation
    """
    
    # ADD CODE HERE (mount b, do not forget Dirichlet boundary conditions!)
    
    x = scipy.sparse.linalg.spsolve(A,b)
    
    return x 

def evolveDiffusion2D(u,v,dx,dy,dt,nu,theta):
    """
    Evolve the 2D heat equation
    using theta scheme and Dirichlet boudanry conditions
    at four sides of the 2D domain using operator splitting 
    for x- and y-directions
    """
    
    rowU , colU = u.shape
    rowV , colV = v.shape
    
    un = u.copy()
    vn = v.copy()
    
    ux = u.copy()
    vx = v.copy()
    
    Dx = nu*dt/dx/dx
    Dy = nu*dt/dy/dy
    
    # x-sweep
    
    #mount A for u and then solve along x
     
    #mount A for v and then solve along x
    
    
    # y-sweep
    
    #mount A for u and then solve along y
    
    #mount A for v and then solve along y
    
    return u, v


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
    tEnd = 0.5
    # discretization of space-time domain
    nx = 100    # number of grid points for spatial mesh in x-direction
                 
    dx = (xMax-xMin)/(nx-1) # spatial mesh spacing
    
    ny = 100    # number of grid points for spatial mesh in y-direction
                 
    dy = (yMax-yMin)/(ny-1) # spatial mesh spacing
    
         
    x = np.linspace(xMin,xMax,nx) 
    y = np.linspace(yMin,yMax,ny)

    # solution arrays
    u = np.zeros((ny, nx)) ##create a 1xn vector of 1's
    v = np.zeros((ny, nx))
    un = u.copy()
    vn = v.copy()
    
    # mesh coordinates    
    X, Y = np.meshgrid(x, y)

    # initial condition
    x0 = 1.
    y0 = 1.
    sigma = 0.1 

    u[:,:] = np.exp(-((X-x0)**2+(Y-y0)**2)/2./sigma**2) 
    v[:,:] = np.exp(-((X-x0)**2+(Y-y0)**2)/2./sigma**2)

    # visualization
    plt.figure()
    CS = plt.contour(X, Y, u,20)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('2D Diffusion')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()
    plt.close()
    
    # enter time loop 
    nMax = 10000
    t = tIni
    tic = time.clock()
    for i in range(nMax):
        dt = 1e-2
        if t+dt>tEnd:
            dt = tEnd - t
        if np.abs(t-tEnd)<1e-8:
            break
        
        # evolve u,v
        u, v = evolveDiffusion2D(u,v,dx,dy,dt,nu,0.5)
        # boundary conditions
        u[0, :] = 0.
        u[-1, :] = 0.
        u[:, 0] = 0.
        u[:, -1] = 0.
        
        v[0, :] = 0.
        v[-1, :] = 0.
        v[:, 0] = 0.
        v[:, -1] = 0.
        print "time : %.5e (dt: %.5e) " % (t,dt)
        t += dt
    toc = time.clock()
    print "CPU time %.5e" % (toc-tic)
    plt.figure()
    CS = plt.contour(X, Y, u,20)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('2D Diffusion')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()
        
    print "DONE"
    
    
    
    
