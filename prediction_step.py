import numpy as np

def prediction_step(L, dx, space_discretization_step, D, dt, lower_bound):
    
    """
    Spatial translation of the likelihood L corresponding to the movement dx.
    (2nd order approximation, symmetric)
    """
    
    if np.linalg.norm(dx) > space_discretization_step:
        print('Warning: -dx- is rather large for the prediction step')
    
    space_discretization_step_2 = np.power(space_discretization_step,2)
    
    up = np.roll(L, 1, axis=0)
    down = np.roll(L, -1, axis=0)
    left = np.roll(L, 1, axis=1)
    right = np.roll(L, -1, axis=1)
    up_left = np.roll(up, 1, axis=1)
    down_right = np.roll(down, -1, axis=1)


    # gradient
    grad_x = (down-up) / (2*space_discretization_step)
    grad_y = (right-left) / (2*space_discretization_step)
    # Hessian matrix
    H_xx = (down+up-2*L) / space_discretization_step_2
    H_yy = (right+left-2*L) / space_discretization_step_2
    H_xy = (up_left+down_right+2*L-up-down-left-right) / (2*space_discretization_step_2)

    # 2nd order expansion + translational noise
    L_predict = L + dx[0]*grad_x + dx[1]*grad_y \
    + 0.5*H_xx*np.power(dx[0],2) + 0.5*H_yy*np.power(dx[1],2) + H_xy*dx[0]*dx[1] \
        + D*dt*(H_xx+H_yy)
        
    return L_predict

