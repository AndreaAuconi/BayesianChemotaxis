import numpy as np
import numpy.linalg as algebra
import matplotlib.pyplot as plt


def decision(L,strategy,R_x_ij,r_x_ij,grad_r_x_ij,H_r_x_ij,Laplace_r_x_ij,grad_Laplace_r_x_ij,agent_size):

    N = len(L)
    center = (N-1)/2
    max_L_loc_index = np.unravel_index(L.argmax(), L.shape)

    if strategy == 'Infotaxis':         
        r_mean = np.sum(np.multiply(L,r_x_ij))
        f_x = np.multiply(np.log(r_mean)-np.log(r_x_ij),grad_r_x_ij[:,:,0])
        f_y = np.multiply(np.log(r_mean)-np.log(r_x_ij),grad_r_x_ij[:,:,1])
        g_x = np.multiply(L,f_x)
        g_y = np.multiply(L,f_y)
        move_direction = np.array([np.sum(g_x),np.sum(g_y)])
        norm = algebra.norm(move_direction)
        if norm > 0:
            move_direction_a_0 = move_direction/norm       
        else:
            move_direction_a_0 = np.array([1,0])

        # Here are the 4 terms appearing in the infotaxis decision
        grad_x_mean = np.sum(np.multiply(L,grad_r_x_ij[:,:,0]))
        grad_y_mean = np.sum(np.multiply(L,grad_r_x_ij[:,:,1]))
        Laplace_mean = np.sum(np.multiply(L,Laplace_r_x_ij))
        q1_x = np.multiply(np.log(r_mean)-np.log(r_x_ij),grad_Laplace_r_x_ij[:,:,0])
        q1_y = np.multiply(np.log(r_mean)-np.log(r_x_ij),grad_Laplace_r_x_ij[:,:,1])
        q2_x = np.multiply(Laplace_mean/r_mean-Laplace_r_x_ij/r_x_ij,grad_r_x_ij[:,:,0])
        q2_y = np.multiply(Laplace_mean/r_mean-Laplace_r_x_ij/r_x_ij,grad_r_x_ij[:,:,1])
        q3_x = 2*((grad_x_mean/r_mean-grad_r_x_ij[:,:,0]/r_x_ij)*H_r_x_ij[:,:,0,0]+(grad_y_mean/r_mean-grad_r_x_ij[:,:,1]/r_x_ij)*H_r_x_ij[:,:,0,1])
        q3_y = 2*((grad_x_mean/r_mean-grad_r_x_ij[:,:,0]/r_x_ij)*H_r_x_ij[:,:,1,0]+(grad_y_mean/r_mean-grad_r_x_ij[:,:,1]/r_x_ij)*H_r_x_ij[:,:,1,1])
        q4_x = np.multiply(np.power(algebra.norm(grad_r_x_ij,axis=2)/r_x_ij,2)-(np.power(grad_x_mean,2)+np.power(grad_y_mean,2))/np.power(r_mean,2),grad_r_x_ij[:,:,0])
        q4_y = np.multiply(np.power(algebra.norm(grad_r_x_ij,axis=2)/r_x_ij,2)-(np.power(grad_x_mean,2)+np.power(grad_y_mean,2))/np.power(r_mean,2),grad_r_x_ij[:,:,1])
        q_x = np.sum(np.multiply(L,q1_x+q2_x+q3_x+q4_x))
        q_y = np.sum(np.multiply(L,q1_y+q2_y+q3_y+q4_y))
        correction = np.power(agent_size/2,2)*np.array([q_x,q_y])
    
        move_direction += correction
        norm = algebra.norm(move_direction)
        if norm > 0:
            move_direction /= norm       
        else:
            move_direction = np.array([1,0])


    elif strategy == 'min_dR':
        up = np.roll(L, 1, axis=0)
        down = np.roll(L, -1, axis=0)
        left = np.roll(L, 1, axis=1)
        right = np.roll(L, -1, axis=1)
        move_direction = np.array([-np.sum(np.multiply(down-up,R_x_ij)),-np.sum(np.multiply(right-left,R_x_ij))])
        norm = algebra.norm(move_direction)
        if norm > 0:
            move_direction /= norm       
        else:
            move_direction = 1
        move_direction_a_0 = move_direction

                              
    elif strategy == 'max_L':
        move_direction = np.array([max_L_loc_index[0]-center,max_L_loc_index[1]-center])
        norm = algebra.norm(move_direction)
        if norm > 0:
            move_direction /= norm       
        else:
            move_direction = np.array([1,0])
        move_direction_a_0 = move_direction
            
    else:
        print('Please specify a strategy')
        move_direction = np.array([0,0])

        
    return move_direction,max_L_loc_index,move_direction_a_0




def plot_decision(L,r_x_ij,grad_r_x_ij,H_r_x_ij,Laplace_r_x_ij,grad_Laplace_r_x_ij,agent_size,move_direction,move_direction_a_0,max_L_loc_index,l,filename):

    r_mean = np.sum(np.multiply(L,r_x_ij))
    grad_abs = np.sqrt(np.sum(np.power(grad_r_x_ij[:,:],2),axis=2))
    f = - np.multiply(np.log(r_mean)-np.log(r_x_ij),grad_abs)
    g = np.multiply(L,f)
    N = len(L)
    center = (N-1)/2
    max_L_direction = np.array([max_L_loc_index[0]-center,max_L_loc_index[1]-center])
    norm = algebra.norm(max_L_direction)
    max_L_direction /= norm 
    
    grad_x_mean = np.sum(np.multiply(L,grad_r_x_ij[:,:,0]))
    grad_y_mean = np.sum(np.multiply(L,grad_r_x_ij[:,:,1]))
    Laplace_mean = np.sum(np.multiply(L,Laplace_r_x_ij))
    q1_x = np.multiply(np.log(r_mean)-np.log(r_x_ij),grad_Laplace_r_x_ij[:,:,0])
    q1_y = np.multiply(np.log(r_mean)-np.log(r_x_ij),grad_Laplace_r_x_ij[:,:,1])
    q2_x = np.multiply(Laplace_mean/r_mean-Laplace_r_x_ij/r_x_ij,grad_r_x_ij[:,:,0])
    q2_y = np.multiply(Laplace_mean/r_mean-Laplace_r_x_ij/r_x_ij,grad_r_x_ij[:,:,1])
    q3_x = 2*((grad_x_mean/r_mean-grad_r_x_ij[:,:,0]/r_x_ij)*H_r_x_ij[:,:,0,0]+(grad_y_mean/r_mean-grad_r_x_ij[:,:,1]/r_x_ij)*H_r_x_ij[:,:,0,1])
    q3_y = 2*((grad_x_mean/r_mean-grad_r_x_ij[:,:,0]/r_x_ij)*H_r_x_ij[:,:,1,0]+(grad_y_mean/r_mean-grad_r_x_ij[:,:,1]/r_x_ij)*H_r_x_ij[:,:,1,1])
    q4_x = np.multiply(np.power(algebra.norm(grad_r_x_ij,axis=2)/r_x_ij,2)-(np.power(grad_x_mean,2)+np.power(grad_y_mean,2))/np.power(r_mean,2),grad_r_x_ij[:,:,0])
    q4_y = np.multiply(np.power(algebra.norm(grad_r_x_ij,axis=2)/r_x_ij,2)-(np.power(grad_x_mean,2)+np.power(grad_y_mean,2))/np.power(r_mean,2),grad_r_x_ij[:,:,1])
    pre_q_x = np.multiply(L,q1_x+q2_x+q3_x+q4_x)
    pre_q_y = np.multiply(L,q1_y+q2_y+q3_y+q4_y)
    g_correction = np.power(agent_size/2,2)*algebra.norm(np.array([pre_q_x,pre_q_y]),axis=0)
    
    g += g_correction  

   
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(np.transpose(np.flip(g,1)),interpolation='None', extent = (-l,l,-l,l))
    title1 = 'Decision incentive'
    ax1.set_title(title1)
    ax1.set_xlim(-0.5*l,0.5*l)
    ax1.set_ylim(-0.5*l,0.5*l)
    ax1.arrow(0,0,max_L_direction[0]*0.1*l,max_L_direction[1]*0.1*l, color='orange', width = 0.005)
    ax1.arrow(0,0,move_direction_a_0[0]*0.1*l,move_direction_a_0[1]*0.1*l, color='brown', width = 0.005)
    ax1.arrow(0,0,move_direction[0]*0.1*l,move_direction[1]*0.1*l, color='green', width = 0.005)
    ax2.imshow(np.transpose(np.flip(L,1)),interpolation='None', extent = (-l,l,-l,l))
    ax2.set_xlim(-0.5*l,0.5*l)
    ax2.set_ylim(-0.5*l,0.5*l)
    ax2.arrow(0,0,max_L_direction[0]*0.1*l,max_L_direction[1]*0.1*l, color='orange', width = 0.005)
    ax2.arrow(0,0,move_direction_a_0[0]*0.1*l,move_direction_a_0[1]*0.1*l, color='brown', width = 0.005)
    ax2.arrow(0,0,move_direction[0]*0.1*l,move_direction[1]*0.1*l, color='green', width = 0.005)
    title2 = 'Likelihood'
    ax2.set_title(title2)
    
    fig.subplots_adjust(hspace=.5)
    
    plt.savefig(filename)
    
    return 0

