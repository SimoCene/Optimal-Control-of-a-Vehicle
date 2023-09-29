# Dynamics function in numerical way with Numpy
# Discrete-time nonlinear dynamics
# Group 19
# Bologna, 17/01/23
#

import numpy as np

# # Dynamics parameters
mm = 1480   #mass [Kg]
Iz = 1950   #inertia[Kgm2]
aa = 1.421  #a  [m]
bb = 1.029  #b  [m]
mu = 1      #mu [nodim]
gg = 9.81   #gravity [m/s^2]



def dynamics(xx,uu,dt):
    """
    Nonlinear dynamics of a car

    Args
        - xx \in \R^6 state at time t
        - uu \in \R^2 input at time t

    Return 
        - next state xx_{t+1}
        - Jacobian of f wrt x, at xx,uu --> AA
        - Jacobian of f wrt u, at xx,uu --> BB

    """
    ############ - Importa valori - #######

    xx = xx[:,None] 
    uu = uu[:,None]

    # x0 = xx[0,0]
    # x1 = xx[1,0]
    # x2 = xx[2,0]
    # x3 = xx[3,0]
    # x4 = xx[4,0]
    # x5 = xx[5,0]

    # u0 = uu[0,0]
    # u1 = uu[1,0]


    x0 = xx[0]
    x1 = xx[1]
    x2 = xx[2]
    x3 = xx[3]
    x4 = xx[4]
    x5 = xx[5]

    u0 = uu[0]
    u1 = uu[1]


    ############ - Equazioni di stati discretizzate con eulero in avanti - #######
    xxP = np.array([
            dt*(x3*np.cos(x2) - x4*np.sin(x2)) + x0, 
            dt*(x3*np.sin(x2) + x4*np.cos(x2)) + x1, 
            dt*x5 + x2, 
            dt*(x4*x5 + (-bb*gg*mm*mu*(u0 - (aa*x5 + x4)/x3)*np.sin(u0)/(aa + bb) + u1*np.cos(u0))/mm) + x3, 
            dt*(-x3*x5 + (-aa*gg*mm*mu*(bb*x5 + x4)/(x3*(aa + bb)) + bb*gg*mm*mu*(u0 - (aa*x5 + x4)/x3)*np.cos(u0)/(aa + bb) + u1*np.sin(u0))/mm) + x4, 
            x5 + dt*(aa*bb*gg*mm*mu*(bb*x5 + x4)/(x3*(aa + bb)) + aa*(bb*gg*mm*mu*(u0 - (aa*x5 + x4)/x3)*np.cos(u0)/(aa + bb) + u1*np.sin(u0)))/Iz]
            , dtype=float)   

    # f derivata in x. 'AA'
    fx = np.array([
            [1, 0, dt*(-x3*np.sin(x2) - x4*np.cos(x2)), dt*np.cos(x2), -dt*np.sin(x2), 0],
            [0, 1, dt*(x3*np.cos(x2) - x4*np.sin(x2)), dt*np.sin(x2), dt*np.cos(x2), 0], 
            [0, 0, 1, 0, 0, dt], 
            [0, 0, 0, -bb*dt*gg*mu*(aa*x5 + x4)*np.sin(u0)/(x3**2*(aa + bb)) + 1, dt*(bb*gg*mu*np.sin(u0)/(x3*(aa + bb)) + x5), dt*(aa*bb*gg*mu*np.sin(u0)/(x3*(aa + bb)) + x4)], 
            [0, 0, 0, dt*(-x5 + (aa*gg*mm*mu*(bb*x5 + x4)/(x3**2*(aa + bb)) + bb*gg*mm*mu*(aa*x5 + x4)*np.cos(u0)/(x3**2*(aa + bb)))/mm), dt*(-aa*gg*mm*mu/(x3*(aa + bb)) - bb*gg*mm*mu*np.cos(u0)/(x3*(aa + bb)))/mm + 1, dt*(-x3 + (-aa*bb*gg*mm*mu*np.cos(u0)/(x3*(aa + bb)) - aa*bb*gg*mm*mu/(x3*(aa + bb)))/mm)], 
            [0, 0, 0, dt*(aa*bb*gg*mm*mu*(aa*x5 + x4)*np.cos(u0)/(x3**2*(aa + bb)) - aa*bb*gg*mm*mu*(bb*x5 + x4)/(x3**2*(aa + bb)))/Iz, dt*(-aa*bb*gg*mm*mu*np.cos(u0)/(x3*(aa + bb)) + aa*bb*gg*mm*mu/(x3*(aa + bb)))/Iz, 1 + dt*(-aa**2*bb*gg*mm*mu*np.cos(u0)/(x3*(aa + bb)) + aa*bb**2*gg*mm*mu/(x3*(aa + bb)))/Iz]
            ], dtype=float) 

    
    # # f derivata in x. 'BB'
    fu = np.array([
            [0, 0], 
            [0, 0], 
            [0, 0], 
            [dt*(-bb*gg*mm*mu*(u0 - (aa*x5 + x4)/x3)*np.cos(u0)/(aa + bb) - bb*gg*mm*mu*np.sin(u0)/(aa + bb) - u1*np.sin(u0))/mm, dt*np.cos(u0)/mm], 
            [dt*(-bb*gg*mm*mu*(u0 - (aa*x5 + x4)/x3)*np.sin(u0)/(aa + bb) + bb*gg*mm*mu*np.cos(u0)/(aa + bb) + u1*np.cos(u0))/mm, dt*np.sin(u0)/mm], 
            [aa*dt*(-bb*gg*mm*mu*(u0 - (aa*x5 + x4)/x3)*np.sin(u0)/(aa + bb) + bb*gg*mm*mu*np.cos(u0)/(aa + bb) + u1*np.cos(u0))/Iz, aa*dt*np.sin(u0)/Iz]
            ], dtype=float)
    
    xxP = xxP.squeeze()

    return xxP, fx, fu