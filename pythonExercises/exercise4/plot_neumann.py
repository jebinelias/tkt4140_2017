def plot_SurfaceNeumann_xy(Temp, Ttop, Tright, Tleft, xmax, ymax, Nx, Ny, nxTicks=4, nyTicks=4):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    # surfaceplot:
    x = np.linspace(0, xmax, Nx + 2)
    y = np.linspace(0, ymax, Ny + 1)
    
    X, Y = np.meshgrid(x, y)
    
    T = np.zeros_like(X)
    
    T[-1,:] = Ttop
    T[:,-1] = Tright
    T[:,0] = Tleft
    k = 1
    for j in range(Ny):
        T[j,1:-1] = Temp[Nx*(k-1):Nx*k]
        k+=1

    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, T, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(0, Ttop)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('T [$^o$C]')
    
    xticks=np.linspace(0.0,xmax,nxTicks+1)
    ax.set_xticks(xticks)
    
    yticks=np.linspace(0.0,ymax,nyTicks+1)
    ax.set_yticks(yticks)