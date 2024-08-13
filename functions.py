import numpy as np
import numpy.linalg as algebra

def field(lambda_,R):
    return lambda_/R

def grad_field(lambda_,x_ij):
    R = algebra.norm(x_ij)
    return -lambda_*x_ij/np.power(R,3)#element-wise division

def H_field(lambda_,x_ij):
    R = algebra.norm(x_ij)
    Hxx = -lambda_/np.power(R,3)+3*lambda_*np.power(x_ij[0],2)/np.power(R,5)
    Hyy = -lambda_/np.power(R,3)+3*lambda_*np.power(x_ij[1],2)/np.power(R,5)
    Hxy = 3*lambda_*x_ij[0]*x_ij[1]/np.power(R,5)
    return np.matrix([[Hxx,Hxy],[Hxy,Hyy]])

def grad_Laplace_field(lambda_,x_ij):
    R = algebra.norm(x_ij)
    return -3*lambda_*x_ij/np.power(R,5)


def r_field(N,space_discretization_step,lambda_,bound):
    
    """
    Matrix of sensing rate r at agent location (center)
    if the source were placed at x_ij
    """
          
    r_x_ij = np.zeros(shape=(N,N), dtype=float)
    center = (N-1)/2
    
    for i in range(N):
        for j in range(N):            
            x_ij = np.array([(i-center)*space_discretization_step,(j-center)*space_discretization_step])
            R = algebra.norm(x_ij)
            if R > bound:
                r_x_ij[i][j] = field(lambda_,R)
            else:
                r_x_ij[i][j] = field(lambda_,bound)
                
    return r_x_ij

def distances_ij(N,space_discretization_step):
    
    """
    Matrix of distances from center to x_ij
    i=0 corrseponds to x=-l+space_discretization_step/2
    i=N-1 corrseponds to x=l-space_discretization_step/2
    """
          
    R_ij = np.zeros(shape=(N,N), dtype=float)
    center = (N-1)/2
    
    for i in range(N):
        for j in range(N):            
            x_ij = np.array([(i-center)*space_discretization_step,(j-center)*space_discretization_step])
            R_ij[i][j] = algebra.norm(x_ij)
                
    return R_ij


def L0(N,space_discretization_step,l):
    
    '''
    initialization of the likelihood.
    '''
    
    polarized = 0.003 #Gaussian shape, polarization is the inverse variance
    #caution, polarized = 0 gives uniform prior, but numerical problem with convection
          
    L = np.zeros(shape=(N,N), dtype=float)
    center = (N-1)/2
    
    for i in range(N):
        for j in range(N):            
            x_ij = np.array([(i-center)*space_discretization_step,(j-center)*space_discretization_step])
            R = algebra.norm(x_ij)
            L[i][j] = np.exp(-polarized*np.power(R,2)/l)
     
    L /= np.sum(L)
           
    return L


def grad_r_field(N,space_discretization_step,lambda_,bound):
         
    grad_r_x_ij = np.zeros(shape=(N,N,2), dtype=float)
    center = (N-1)/2
    
    for i in range(N):
        for j in range(N):            
            x_ij = np.array([(i-center)*space_discretization_step,(j-center)*space_discretization_step])
            R = algebra.norm(x_ij)
            if R > bound:
                grad_r_x_ij[i][j] = grad_field(lambda_,x_ij)
            else:
                grad_r_x_ij[i][j] = grad_field(lambda_,np.array([bound,bound]))
                
    return grad_r_x_ij


def H_r_field(N,space_discretization_step,lambda_,bound):
  
    H_r_x_ij = np.zeros(shape=(N,N,2,2), dtype=float)
    Laplace = np.zeros(shape=(N,N), dtype=float)
    center = (N-1)/2
    
    for i in range(N):
        for j in range(N):            
            x_ij = np.array([(i-center)*space_discretization_step,(j-center)*space_discretization_step])
            R = algebra.norm(x_ij)
            if R > bound:
                H_r_x_ij[i][j] = H_field(lambda_,x_ij)
                Laplace[i][j] = H_r_x_ij[i][j][0][0] + H_r_x_ij[i][j][1][1]
            else:
                H_r_x_ij[i][j] = H_field(lambda_,np.array([bound,bound]))
                Laplace[i][j] = 0
                
    return H_r_x_ij, Laplace

def grad_Laplace_r_field(N,space_discretization_step,lambda_,bound):
         
    grad_Laplace_r_x_ij = np.zeros(shape=(N,N,2), dtype=float)
    center = (N-1)/2
    
    for i in range(N):
        for j in range(N):            
            x_ij = np.array([(i-center)*space_discretization_step,(j-center)*space_discretization_step])
            R = algebra.norm(x_ij)
            if R > bound:
                grad_Laplace_r_x_ij[i][j] = grad_Laplace_field(lambda_,x_ij)
            else:
                grad_Laplace_r_x_ij[i][j] = grad_Laplace_field(lambda_,np.array([bound,bound]))
                
    return grad_Laplace_r_x_ij


def E_dS_components(L,N,space_discretization_step,lambda_,bound):
  
    H_r_x_ij = np.zeros(shape=(N,N,2,2), dtype=float)
    Laplace = np.zeros(shape=(N,N), dtype=float)
    center = (N-1)/2
    
    for i in range(N):
        for j in range(N):            
            x_ij = np.array([(i-center)*space_discretization_step,(j-center)*space_discretization_step])
            R = algebra.norm(x_ij)
            if R > bound:
                H_r_x_ij[i][j] = H_field(lambda_,x_ij)
                Laplace[i][j] = H_r_x_ij[i][j][0][0] + H_r_x_ij[i][j][1][1]
            else:
                H_r_x_ij[i][j] = H_field(lambda_,np.array([bound,bound]))
                Laplace[i][j] = 0
                
    return H_r_x_ij, Laplace
      

def generate_source_position(init_dist):
    psi_init = np.random.uniform(0,2*np.pi)
    R_init = init_dist
    source = np.array([R_init*np.cos(psi_init),R_init*np.sin(psi_init)])
    return source


def entropy(L):
    S = -np.sum(np.multiply(L,np.log(L)))
    return S


  

    