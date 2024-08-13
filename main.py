import numpy as np
import numpy.linalg as algebra
from scipy import stats
from prediction_step import prediction_step
from decision import decision, plot_decision
from functions import L0, generate_source_position, field,\
 distances_ij, r_field, grad_r_field, H_r_field,\
     grad_Laplace_r_field, entropy
import seaborn as sns    
import matplotlib.pyplot as plt
from video import video


# Simulation parameters
dt = 5e-4
sampling_interval = 100 # for saving results and DECISION MAKING
T = 10000 # integration time
N = 200 # number of bins per dimension: space discretization


# NOTE: N << l*np.sqrt(8/(D*dt)) 
# NOTE: N << 4*l/(velocity*dt) 

# Model parameters
#strategy = 'Infotaxis' # 'max_L' 
lambda_ = 2 # field strength, binding rate at distance 1
velocity = 0.01
D = 2.5e-4 # motility noise, translational
agent_size = 0.00 # < space_discretization_step 
max_distance = 0.87
min_distance = 0.03


# estimation parameters
l = 1 # space length
t_init = 50
init_dist = l*0.2
lower_bound = 1e-7/np.power(N,2) # lowest allowed likelihood
space_discretization_step = 2*l/N
bound = 0.5*space_discretization_step# minimum distance where field is computed, for numerical stability
N_angles = 30 #number of receptors, normalized

make_video = True
video_length = 300.0
 

R_t_strategies = []# target distances time series
strategies = ['Infotaxis', 'max_L', 'min_dR']


for strategy in strategies:      
          
    # initialize simulation
    source = generate_source_position(init_dist)
    agent_step_size = velocity*dt
    R_x_ij = distances_ij(N,space_discretization_step)
    r_x_ij = r_field(N,space_discretization_step,lambda_,bound)
    grad_r_x_ij = grad_r_field(N,space_discretization_step,lambda_,bound)
    H_r_x_ij, Laplace_r_x_ij = H_r_field(N,space_discretization_step,lambda_,bound)
    grad_Laplace_r_x_ij = grad_Laplace_r_field(N,space_discretization_step,lambda_,bound)
    L = L0(N,space_discretization_step,l)#np.ones(shape=(N,N), dtype=float)
    min_L = np.ones_like(L) * lower_bound
    angles = [i*2*np.pi/N_angles for i in range(N_angles)]
    directions = [np.array([np.cos(phi),np.sin(phi)]) for phi in angles]
    if make_video == True:
        L_t = []
        source_t = [[],[]]
        max_L_t = [[],[]]
        move_direction_t = []
    
    sampling = 0    
    S_t = []
    R_t = []
    list_t = []#list of times
    
    #expected contributions time series
    info_erasure = []
    concentration_sensing = []
    curvature_correction = []
    gradient_sensing = []
    
    angle_difference = []# for analysis of agent size impact on decision angle
    
    plot_done = [False]*10
    this_plot = 1 

    move_direction, max_L_loc_index, move_direction_a_0 = decision(L,strategy,R_x_ij,r_x_ij,grad_r_x_ij,H_r_x_ij,Laplace_r_x_ij,grad_Laplace_r_x_ij,agent_size)
      
     
    for t in np.arange(0,T,dt):
        
        distance = algebra.norm(source)

        if distance > max_distance:
            source *= 0.995
        if distance < min_distance:
            source *= 1.005
            
        sampling += 1
    
    
        if sampling == sampling_interval:
            move_direction, max_L_loc_index, move_direction_a_0 = decision(L,strategy,R_x_ij,r_x_ij,grad_r_x_ij,H_r_x_ij,Laplace_r_x_ij,grad_Laplace_r_x_ij,agent_size)
            print('progress:',str(100*t/T)[:5],'%')
            sampling = 0
            if t>10:
                S=entropy(L)
                S_t.append(S)
                R_t.append(distance)
                angle_difference.append(np.arctan2(move_direction[1],move_direction[0])-np.arctan2(move_direction_a_0[1],move_direction_a_0[0]))
                list_t.append(t) 
                if make_video == True and t<video_length:
                    L_t.append(L)
                    source_t[0].append(source[0])
                    source_t[1].append(source[1])
                    max_L_t[0].append(max_L_loc_index[0])
                    max_L_t[1].append(max_L_loc_index[1])
                    move_direction_t.append(move_direction)
                print('entropy =',str(S)[:4])
                print('distance / l = ', str(distance/l)[:4])
            
            if strategy == 'Infotaxis':
                if t>(0.1*this_plot*T) and plot_done[this_plot] == False:               
                    filename = 'Decision_' + strategy  + str(lambda_)[:6] + '_' + str(agent_size)[:7] + '_t_' + str(t)[:6] + '.pdf'
                    plot_decision(L,r_x_ij,grad_r_x_ij,H_r_x_ij,Laplace_r_x_ij,grad_Laplace_r_x_ij,agent_size,move_direction,move_direction_a_0,max_L_loc_index,l,filename)
                    plot_done[this_plot] = True
                    this_plot +=1
    
        
        dx = agent_step_size * move_direction
        source -= dx
        source += np.array([np.random.normal(),np.random.normal()]) * np.sqrt(2*D*dt)
    
        
        L_predict = prediction_step(L,dx,space_discretization_step,D,dt,lower_bound)
        
        r_mean = np.sum(np.multiply(L_predict,r_x_ij))
        mean_grad_r = np.array([np.sum(np.multiply(L_predict,grad_r_x_ij[:,:,0])),np.sum(np.multiply(L_predict,grad_r_x_ij[:,:,1]))])
        Laplace_mean = np.sum(np.multiply(L_predict,Laplace_r_x_ij))
 
        if t>t_init and sampling == 0:
            info_erasure.append((entropy(L_predict)-entropy(L))/dt)
            concentration_sensing.append(-(r_mean*np.log(r_mean) -np.sum(np.multiply(L_predict,r_x_ij*np.log(r_x_ij)))))
            curvature_correction.append(-np.power(agent_size/2,2)*(Laplace_mean*np.log(r_mean) -np.sum(np.multiply(L_predict,Laplace_r_x_ij*np.log(r_x_ij)))))
            gradient_sensing.append(-np.power(agent_size/2,2)*(np.power(algebra.norm(mean_grad_r),2)/r_mean -np.sum(np.multiply(L_predict,np.power(algebra.norm(grad_r_x_ij,axis=2),2)/r_x_ij))))            
            
        dm = np.array([0.,0.]) #events
        for phi_direction in directions:
            this_rate = field(lambda_,algebra.norm(source-agent_size*phi_direction))/N_angles                     
            dm += phi_direction * np.random.poisson(this_rate*dt)
 
        if algebra.norm(dm)>1:
            print('Warning, ||dm|| =',algebra.norm(dm))
               
        L = L_predict + L_predict*(r_mean-r_x_ij)*dt + L_predict*np.power(agent_size/2,2)*(Laplace_mean-Laplace_r_x_ij)*dt
        if algebra.norm(dm)>0: 
            L += L_predict*(r_x_ij/r_mean-1)*algebra.norm(dm) + L_predict*agent_size*(r_x_ij/r_mean)*(np.dot(mean_grad_r,dm)/r_mean-np.dot(grad_r_x_ij,dm)/r_x_ij) \
                + L_predict*0.5*np.power(agent_size,2)*(r_x_ij/r_mean)*(np.dot(np.dot(H_r_x_ij,dm),dm)/r_x_ij) \
                   + L_predict*np.power(agent_size,2)*(r_x_ij/r_mean)*(np.dot(mean_grad_r,dm)/r_mean)*(np.dot(mean_grad_r,dm)/r_mean-np.dot(grad_r_x_ij,dm)/r_x_ij)


        L = np.maximum(min_L,L)               
        L /= np.sum(L)
    

         
        
    R_t_strategies.append(R_t)    
    
    mean_R = np.mean(R_t)
    std_err_R = np.std(R_t)/np.sqrt(len(R_t))   
    
    S_t = stats.zscore(S_t)
    R_t = stats.zscore(R_t)

    plt.clf()
    plt.plot(list_t,S_t,linewidth=0.02)
    plt.plot(list_t,R_t,linewidth=0.02)
    plt.legend([r'$S_t$', r'$R_t$'])
    plt.xlabel(r'$t$')
    namefile = 'entropy_dynamics_' + strategy + str(lambda_)[:7] + '_' + str(agent_size)[:7] + '.pdf'
    plt.savefig(namefile)

    plt.clf()
    plt.plot(list_t,angle_difference,linewidth=0.02)


    plt.clf()
    sns.kdeplot(angle_difference, shade=True, clip=(-2*np.std(angle_difference), 2*np.std(angle_difference)))
    plt.xlabel(r'$\Delta\phi$')
    title = r'$\langle(\Delta\phi)^2\rangle^{0.5} =$' + str(np.std(angle_difference))[:8]
    plt.title(title)
    namefile = 'angle_deviation_' + strategy + str(lambda_)[:7] + '_' + str(agent_size)[:7] + '.pdf'
    plt.savefig(namefile)    

    plt.clf()
    sns.kdeplot(R_t, shade=True, color="forestgreen")
    sns.kdeplot(S_t, shade=True, color="orange")
    plt.legend([r"$R$", r"$S$"],loc='best')
    title = "avg(R) = " + str(mean_R)[:5] + "+/-" + str(std_err_R)[:5]
    plt.title(title)
    namefile = 'Hist_' + strategy + str(lambda_)[:7] +  '_' + str(agent_size)[:7] + '.pdf'
    plt.savefig(namefile)
    
    
    plt.clf()
    sns.kdeplot(info_erasure, shade=True, log_scale = True)
    sns.kdeplot(concentration_sensing, shade=True, log_scale = True)
    sns.kdeplot(gradient_sensing, shade=True, log_scale = True)
    sns.kdeplot(curvature_correction, shade=True, log_scale = True)
    plt.legend(['motility noise','concentration sensing','gradient sensing','Laplacian correction'],loc='upper left')
    plt.xlabel(r'$\langle dS\rangle/dt$')
    plt.ylabel(r'$p(\langle dS\rangle /dt)$')
    title = r'$\langle dS\rangle/dt$' + ' components; ' + r'$\lambda = $' + str(lambda_)[:5] + r'; $a = $' + str(agent_size)[:7]
    plt.title(title)
    namefile = 'Components_E_dS_' + strategy + str(lambda_)[:7] + '_' + str(agent_size)[:7] + '.pdf'
    plt.savefig(namefile)
    
    if agent_size>0:
        plt.clf()
        sns.histplot(info_erasure, log_scale = True, color = 'red', fill = False)
        sns.histplot(concentration_sensing, log_scale = True, color = 'blue', fill = False)
        sns.histplot(gradient_sensing, log_scale = True, color = 'green', fill = False)
        sns.histplot(curvature_correction, log_scale = True, color = 'yellow', fill = False)
        plt.legend(['motility noise','concentration sensing','gradient sensing','Laplacian correction'],loc='upper left')
        plt.xlabel(r'$\langle dS\rangle/dt$')
        plt.ylabel(r'$p(\langle dS\rangle /dt)$')
        title = r'$\langle dS\rangle/dt$' + ' components; ' + r'$\lambda = $' + str(lambda_)[:5] + r'; $a = $' + str(agent_size)[:7]
        plt.title(title)
        namefile = 'HIST_Components_E_dS_' + strategy + str(lambda_)[:7] + '_' + str(agent_size)[:7] + '.pdf'
        plt.savefig(namefile)
   

    if make_video == True:    
        namefile = strategy + str(lambda_)[:5] + '_' + str(agent_size)[:7] + '.mp4'
        video(L_t, source_t, max_L_t, move_direction_t, l, N, namefile)
 


plt.clf()
for i in range(len(strategies)):
    sns.kdeplot(R_t_strategies[i], shade=True, clip=(0.0, 0.85*l))
plt.legend(strategies,loc='best')
plt.xlabel(r'$R$')
plt.ylabel(r'$p(R)$')
title = r'$\lambda = $' + str(lambda_)[:5] + r'; $a = $' + str(agent_size)[:7]
plt.title(title)
namefile = 'Hist_strategies_' + str(lambda_)[:7] + '_' + str(agent_size)[:7] + '.pdf'
plt.savefig(namefile)
       
