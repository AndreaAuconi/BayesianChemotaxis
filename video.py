import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def video(L_t, source_t, max_L_t, move_direction_t, l, N, namefile):
    
    print('producing video .....')
    
    plt.clf()
   
    fig = plt.figure()
    
    data_L = np.transpose(np.flip(L_t[3],1))
    plot_likelihood = plt.imshow(data_L, interpolation='None', extent = (-l,l,-l,l))
    plot_source = plt.scatter([], [], color='r', s = 7)
    plot_arrow = plt.scatter([], [], color='green', s = 1)
    plt.xlim(-0.87*l,0.87*l)
    plt.ylim(-0.87*l,0.87*l)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter([0], [0], color='pink', s = 7)

    
    
    def init():
        
        plot_likelihood.set_data(np.zeros((N,N)))
        plot_source.set_offsets([0,0])
        plot_arrow.set_offsets([0,0])
      
        return plot_likelihood, plot_source, plot_arrow
    
    
    def animate(i):
           
        plot_likelihood.set_data(np.transpose(np.flip(L_t[i],1)))
        the_source = [source_t[0][i], source_t[1][i]]
        plot_source.set_offsets([the_source[0], the_source[1]])
        #max_L = [(max_L_t[0][i]-center)*space_discretization_step,(max_L_t[1][i]-center)*space_discretization_step]
        arrow_data = [[move_direction_t[i][0]*0.1*l*j/10,move_direction_t[i][1]*0.1*l*j/10] for j in range(8)]
        plot_arrow.set_offsets(arrow_data)
            
        return plot_likelihood, plot_source, plot_arrow
    
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, interval = 1, frames = len(L_t)-1)
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=100 , metadata=dict(artist='Me')) 
    anim.save(namefile, writer=writer)
    
    print('done!')

