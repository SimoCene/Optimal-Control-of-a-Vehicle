# Initial Trajectory function
# Group 19
# Bologna, 24/01/2023
#

import numpy as np
import Dynamics_Numpy as dyn
import Reference_curve as Ref
Test_function = False

######################################
# Initial Trajectory - Line
######################################

def Line_Trj(ns,ni,TT,dt,V0):
    # V0 = 1  # Define the costant velociy

    xx_initial = np.zeros((ns, TT))
    uu_initial = np.zeros((ni, TT))

    for tt in range(0, TT):
        xx_initial[0,tt] = V0*dt*tt # X 
        xx_initial[1,tt] = 0        # Y
        xx_initial[2,tt] = 0        # Phi
        xx_initial[3,tt] = V0       # Vx
        xx_initial[4,tt] = 0        # Vy
        xx_initial[5,tt] = 0        # Phi_dot
    
    for tt in range(0, TT):
        uu_initial[0,tt] = 0        # Delta
        uu_initial[1,tt] = 0        # Fx

    return xx_initial, uu_initial


def PID_Trj_1(ns,ni,TT,dt,xx_ref):
    """
    Porportional controller on the xx_ref for the Skidpad
    Stearing angle proportional on the error of the Phi
    Force proportional on the error of the Vx [Is the acc in discrate time] and the mass (1460)
    """
    xx = np.zeros((ns, TT))   # state seq.
    uu = np.zeros((ni, TT))   # input seq.
    xx[:,0] = xx_ref[:,0]   #Save the initial state
    # xx[3,0] = xx_ref[3,0]
    # xx[2,0] = np.pi*0.5 # yaw iniziale a 90°

    for tt in range(TT-1):
        uu[0,tt] = 1 * xx_ref[2,tt]-xx[2,tt]
        uu[1,tt] = 1460 * (xx_ref[3,tt]-xx[3,tt])
        xx[:,tt+1] = dyn.dynamics(xx[:,tt], uu[:,tt], dt)[0]

    # Così ottengo una traiettoria che però ha uno statio iniziale diverso da quella dalla curva di riferimento, quindi inizialmente il costo del Newton salirà
    # xx[1,0] = xx[1,0]-2.1   #Modifica lo stato iniziale shiftato sulla Y di -2.1. Così ottengo un curva.
    # for tt in range(TT-1):  #Dato la modifca dello stato utilizzando gli stessi input riproduca una traiettoria
    #     xx[:,tt+1] = dyn.dynamics(xx[:,tt], uu[:,tt], dt)[0]
    
    uu[:,-1] = uu[:,-2] # for plot, because you don't have the last value
    return xx, uu


def PID_Trj_2(ns,ni,TT,dt,xx_ref,Gain=140):
    """
    Porportional controller on the xx_ref for the Skidpad
    Stearing angle proportional on the error of Phi and K_theta
    Force proportional on the error of posistion and K_force = 100 * K_theta
    """
    xx = np.zeros((ns, TT))   # state seq.
    uu = np.zeros((ni, TT))   # input seq.
    xx[:,0] = xx_ref[:,0]

    ee = np.zeros(TT)
    k_theta = Gain/100
    k_force = Gain

    for tt in range(TT-1):
        uu[0,tt] = k_theta * (xx_ref[2,tt]-xx[2,tt])
        ee[tt] = ((xx_ref[0,tt]-xx[0,tt])**2 + (xx_ref[1,tt]-xx[1,tt])**2)**(1/2) #Errore di posizione al tempo tt
        uu[1,tt] = k_force * ee[tt]
        xx[:,tt+1] = dyn.dynamics(xx[:,tt], uu[:,tt], dt)[0]

    uu[:,-1] = uu[:,-2] # for plot, because you don't have the last value
    return xx, uu

'''
def P_ctr_Trj_2(ns,ni,TT,dt,xx_ref,k_theta=2,k_force=2200):
    xx = np.zeros((ns, TT))   # state seq.
    uu = np.zeros((ni, TT))   # input seq.

    xx[:,0] = xx_ref[:,0]
    Buco_in_mezzo = 40
    ee = np.zeros(TT+1)   # input seq.

    for tt in range(int(TT/2)-Buco_in_mezzo):
        # ee[tt+1] = ((xx_ref[0,tt]-xx[0,tt])**2 + (xx_ref[1,tt]-xx[1,tt])**2)**(1/2) #Errore al tempo tt
        # if tt<(int(TT/2)) or tt>(int(TT/2)+300):
        uu[0,tt] = k_theta * (xx_ref[5,tt]-xx[5,tt])
        uu[1,tt] = k_force*(xx_ref[3,tt]-xx[3,tt])
        xx[:,tt+1] = dyn.dynamics(xx[:,tt], uu[:,tt], dt)[0]

    for tt in range(int(TT/2)-Buco_in_mezzo, int(TT/2)+Buco_in_mezzo):
        # ee[tt+1] = ((xx_ref[0,tt]-xx[0,tt])**2 + (xx_ref[1,tt]-xx[1,tt])**2)**(1/2) #Errore al tempo tt
        # if tt<(int(TT/2)) or tt>(int(TT/2)+300):
        uu[0,tt] = k_theta * (xx_ref[2,tt]-xx[2,tt])
        uu[1,tt] = k_force*(xx_ref[3,tt]-xx[3,tt])
        xx[:,tt+1] = dyn.dynamics(xx[:,tt], uu[:,tt], dt)[0]

    for tt in range(int(TT/2)+Buco_in_mezzo, TT-1):
        # ee[tt+1] = ((xx_ref[0,tt]-xx[0,tt])**2 + (xx_ref[1,tt]-xx[1,tt])**2)**(1/2) #Errore al tempo tt
        # if tt<(int(TT/2)) or tt>(int(TT/2)+300):
        uu[0,tt] = k_theta * (xx_ref[5,tt]-xx[5,tt])
        uu[1,tt] = k_force*(xx_ref[3,tt]-xx[3,tt])
        xx[:,tt+1] = dyn.dynamics(xx[:,tt], uu[:,tt], dt)[0]

    xx[1,0] = xx[1,0]-2.1

    for tt in range(TT-1):
        xx[:,tt+1] = dyn.dynamics(xx[:,tt], uu[:,tt], dt)[0]

    return xx, uu
'''




#######################################
# Test the function
#######################################
if Test_function:
    import matplotlib.pyplot as plt

    tf = 40   # final time in seconds
    dt = 1e-2    # get discretization step from dynamics
    ns = 6
    ni = 2
    TT = int(tf/dt) # discrete-time samples

    xx_initial = np.zeros((ns, TT))
    uu_initial = np.zeros((ni, TT))
    # xx_initial, uu_initial = Line_Trj(ns,ni,TT,dt, V0=1)

    xx_ref, uu_ref = Ref.Skidpad_Curve(ns, ni, TT, dt)
    xx_initial, uu_initial = PID_Trj_2(ns,ni,TT,dt,xx_ref)

    tt_hor = np.linspace(0,tf,TT)

    fig, axs = plt.subplots(ns+ni, 1, sharex='all')

    axs[0].plot(tt_hor, xx_initial[0,:], 'g--', linewidth=2)
    axs[0].grid()
    axs[0].set_ylabel('$x_0$')

    axs[1].plot(tt_hor, xx_initial[1,:], 'g--', linewidth=2)
    axs[1].grid()
    axs[1].set_ylabel('$x_1$')

    axs[2].plot(tt_hor, xx_initial[2,:], 'g--', linewidth=2)
    axs[2].grid()
    axs[2].set_ylabel('$x_2$')

    axs[3].plot(tt_hor, xx_initial[3,:], 'g--', linewidth=2)
    axs[3].grid()
    axs[3].set_ylabel('$x_3$')

    axs[4].plot(tt_hor, xx_initial[4,:], 'g--', linewidth=2)
    axs[4].grid()
    axs[4].set_ylabel('$x_4$')

    axs[5].plot(tt_hor, xx_initial[5,:], 'g--', linewidth=2)
    axs[5].grid()
    axs[5].set_ylabel('$x_5$')


    axs[6].plot(tt_hor, uu_initial[0,:], 'r--', linewidth=2)
    axs[6].grid()
    axs[6].set_ylabel('$u_0$')

    axs[7].plot(tt_hor, uu_initial[1,:], 'r--', linewidth=2)
    axs[7].grid()
    axs[7].set_ylabel('$u_1$')
    
    # plt.show()




    plt.figure('OTTO')
    plt.plot(xx_ref[0,:], xx_ref[1,:], 'g--', linewidth=2.5, label='$Ref$')
    plt.plot(xx_initial[0,:], xx_initial[1,:], linewidth=2.5, label='$Star$')
    plt.grid()
    plt.ylabel('$Y$')
    plt.xlabel('$X$')
    plt.legend()
    plt.axis('equal')

    plt.title('Skidpad')
    plt.show()

# np.deg2rad(ref_deg_T)

