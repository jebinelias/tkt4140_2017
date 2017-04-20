import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

def getMaxDt(u,v,dx,dy,cfl):
    """
    Compute maximum admissible time step according to the CFL condition
    """
    
    return dtMax
    
def fluxBurger(u,v,fluxType):
    return (1.-fluxType)*u*u+fluxType*u*v
   
@jit   
def evolveBurger1D(u,v,dx,dt,fluxType):
    """
    evolve 1D array u for Burgers equation
    fluxType is related to u*u or u*v fluxes 
    in 2D Burgers
    """
    
             
    return u
    
if __name__ == '__main__':  # this allows for the "main" part of the code to be
                            # executed only if the module is run as a program,
                            # while it allows to import it as a module
                            # in order to reuse functions defined here

    
    # space-time domain
    cfl = 0.9
    xMin = 0.
    xMax = 2.
    yMin = 0.
    yMax = 2.
    tIni = 0.
    tEnd = 0.2
     
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
    
    row , col = u.shape
    
    # mesh coordinates    
    X, Y = np.meshgrid(x, y)

    # initial condition
    x0 = 1.
    y0 = 1.
    sigma = 0.3 

    u[:,:] = np.exp(-((X-x0)**2+(Y-y0)**2)/2./sigma**2) 
    v[:,:] = np.exp(-((X-x0)**2+(Y-y0)**2)/2./sigma**2)

    # visualization
    plt.figure()
    CS = plt.contour(X, Y, u,20)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('2D Burgers')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()
    plt.close()
    
    # enter time loop 
    nMax = 10000
    t = tIni
    tic = time.clock()
    for i in range(nMax):
        # get dt according to CFL condition
        dt = getMaxDt(u,v,dx,dy,cfl)
        # ensure exact matching of tEnd
        if t+dt>tEnd:
            dt = tEnd - t
        # finish simulation if tEnd was reached
        if np.abs(t-tEnd)<1e-8:
            break
        # evolve u,v from t^n to t^{n+1}
        ux = u.copy()
        vx = v.copy()
        
        # x-sweep: solve problems along x-direction
        for i in range(1,row-1):
            ux[i,:] = evolveBurger1D(u[i,:],v[i,:],dx,dt,0.)
            # vx update has to be added 
        # y-sweep: solve problems along y-direction
        # use solution of x-sweep as initial condition!
        # vx update has to be added
        for j in range(1,col-1):
            # u (along y) update has to be added
            # v (along y) update has to be added
        # set boundary conditions
        u[0, :] = 0.
        u[-1, :] = 0.
        u[:, 0] = 0.
        u[:, -1] = 0.
        v[0, :] = 0.
        v[-1, :] = 0.
        v[:, 0] = 0.
        v[:, -1] = 0.
        print "%i ::: time : %.5e (dt: %.5e) " % (i,t,dt)
        # update time
        t += dt
    toc = time.clock()
    print "CPU time %.5e" % (toc-tic)
    # plot result
    plt.figure()
    CS = plt.contour(X, Y, u,20)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('2D Burgers')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.show()

        
    print "DONE"
    
    
    
    
