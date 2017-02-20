# python_exercises/pexercise2/myanimation.py

import matplotlib.pyplot as plt
from math import sin, cos, pi
from matplotlib import animation

def animatePendulum(theta, E, t, l=1):
            
    # set up figure and animation
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    #ax.grid()
    
    line, = ax.plot([], [], '-o', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    theta_text =  ax.text(0.02, 0.85, '', transform=ax.transAxes)
    def init():
        """initialize animation"""
        line.set_data([], [])
        time_text.set_text('')
        energy_text.set_text('')
        theta_text.set_text('')
        return line, time_text, energy_text
    
    def animate(i):
        """perform animation step"""
        x = l*sin(theta[i])
        y = -l*cos(theta[i])
        deg = theta[i]*180/pi
        
        line.set_data([0, x], [0, y])
        time_text.set_text('time = %.1f [-]' % t[i])
        energy_text.set_text('energy = %.5f [-]' % E[i])
        theta_text.set_text(r'$\theta$ = %.0f $^o$' % deg)
        return line, time_text, energy_text
    
    # choose the interval based on dt and the time to animate one step
    Npictures = len(t)
    ani = animation.FuncAnimation(fig, animate,init_func=init,frames=Npictures,interval=1,blit=False)
    

    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #ani.save('pendulum.mov', writer=writer)
    
    plt.show()