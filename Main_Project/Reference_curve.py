# Reference curve function
# Group 19
# Bologna, 24/01/2023
#

import numpy as np
Test_function = False


def Step_Curve(ns,ni,TT,dt,Velocity,Step_value):
    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    for tt in range(0, int(TT/2)):
        xx_ref[0,tt] = Velocity*dt*tt
        xx_ref[1,tt] = 0
        xx_ref[2,tt] = 0
        xx_ref[3,tt] = Velocity
        xx_ref[4,tt] = 0
        xx_ref[5,tt] = 0

    for tt in range(int(TT/2), TT):
        xx_ref[0,tt] = Velocity*dt*tt
        xx_ref[1,tt] = Step_value
        xx_ref[2,tt] = 0
        xx_ref[3,tt] = Velocity
        xx_ref[4,tt] = 0
        xx_ref[5,tt] = 0
    
    for tt in range(0, TT):
        uu_ref[0,tt] = 0
        uu_ref[1,tt] = 0

    return xx_ref, uu_ref


def Sigmoid_Curve(ns,ni,TT,dt,X0,V0,Y0,YT,Scaling_Factor=6):
    """
    X0 - Posizione iniziale
    V0 = 1  # Define the costant velociy
    Y0 - Y tempo iniziale
    YT - Y tempo finale
    ### Errore #### 
    Così è l'angolo dall'origine alla posizione attuale, non l'angolo con l'orizzonte
    xx_ref[2,tt] = np.arctan(xx_ref[1,tt] / xx_ref[0,tt])     
    Arctg(Y/X), da errore ad inizio perchè diviso 0. Per questo va inserito X0 != 0
    """

    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    for tt in range(0, TT):
        tt_Sigmoid = (((tt - TT/2) / (TT/2)) *Scaling_Factor)
        # Troppo grande tt come input. Va centrato a 0 in TT/2.
        # Va anche scalato di /TT/2 per renderlo unitario.
        # Va moltiplicato di *6 perchè la Sigmoid lavora nel range [-6, +6]

        xx_ref[0,tt] = V0*dt*tt + X0
        xx_ref[1,tt] = Y0 + sigmoid_fcn(tt_Sigmoid)[0]*(YT - Y0) 
        xx_ref[2,tt] = sigmoid_fcn(tt_Sigmoid)[1]*(YT - Y0)
        xx_ref[3,tt] = V0
        xx_ref[4,tt] = 0
        xx_ref[5,tt] = sigmoid_fcn(tt_Sigmoid)[2]*(YT - Y0)  #Si potrebbe calcolare XX2_dot in forma analitica. Fatto googlando
    
    for tt in range(0, TT):
        uu_ref[0,tt] = 0
        uu_ref[1,tt] = 1  #F = m*A, in questo caso Vx_dot è nulla

    return xx_ref, uu_ref


def Skidpad_Curve(ns,ni,TT,dt):
  xx_ref = np.zeros((ns, TT))
  uu_ref = np.zeros((ni, TT))

  rr_0 = 15.25/2
  rr_1 = 21.25/2
  rr = (rr_0+rr_1)/2
  XXc_1 = rr
  YYc_1 = 0
  XXc_2 = -rr
  YYc_2 = 0

  #Lo stato iniziale (il primo millesimo) provo a ridurre le velocità per ridurre lo sforzo di controllo
  # Theta_0 = np.pi
  # millesimo = int(TT/1000)
  # for tt in range(0, millesimo):
  #   xx_ref[0,tt] = XXc_1 + rr*np.cos(Theta_0)
  #   xx_ref[1,tt] = YYc_1 + rr*np.sin(Theta_0)
  #   xx_ref[2,tt] = Theta_0-(np.pi/2)
  #   xx_ref[3,tt] = 0.01 # Vx = R * Theta_dot
  #   xx_ref[4,tt] = 0
  #   xx_ref[5,tt] = 0

  #Il primo tratto va parametrizzato Theta_1 [np.pi, -np.pi] --> Orario
  Theta_1 = np.linspace(np.pi, -np.pi, int(TT/2))
  dTheta_1 = Theta_1[3]-Theta_1[2]
  for tt in range(0, int(TT/2)):
    xx_ref[0,tt] = XXc_1 + rr*np.cos(Theta_1[tt])
    xx_ref[1,tt] = YYc_1 + rr*np.sin(Theta_1[tt])
    xx_ref[2,tt] = Theta_1[tt]-(np.pi/2)
    xx_ref[3,tt] = rr*(np.abs(dTheta_1))/dt # Vx = R * Theta_dot
    xx_ref[4,tt] = 0
    xx_ref[5,tt] = dTheta_1/dt

  #Il secondo tratto va parametrizzato Theta_2 [0, 2*np.pi] --> Anti-orario
  Theta_2 = np.linspace(0, 2*np.pi, int(TT/2))
  dTheta_2 = Theta_2[3]-Theta_2[2]
  for tt in range(int(TT/2), TT):
    xx_ref[0,tt] = XXc_2 + rr*np.cos(Theta_2[tt-int(TT/2)])
    xx_ref[1,tt] = YYc_2 + rr*np.sin(Theta_2[tt-int(TT/2)])
    xx_ref[2,tt] = Theta_2[tt-int(TT/2)]-(3/2*np.pi)
    xx_ref[3,tt] = rr*(np.abs(dTheta_2))/dt
    xx_ref[4,tt] = 0
    xx_ref[5,tt] = dTheta_2/dt
  
  for tt in range(0, TT):
    uu_ref[0,tt] = 0
    uu_ref[1,tt] = 0

  return xx_ref, uu_ref

######################################
# Define of sigmoid
######################################

def sigmoid_fcn(tt):
  """
    Sigmoid function

    Return
    - s = 1/1+e^-t
    - ds = d/dx s(t)
    - dds = d/dx ds(t)
  """
  ss = 1/(1 + np.exp(-tt))
  ds = ss*(1-ss)
  dds = ss*(1-ss)*(1-2*ss)

  return ss, ds, dds



#######################################
# Test the function
#######################################
if Test_function:
  import matplotlib.pyplot as plt

  tf = 38   # final time in seconds
  dt = 1e-2    # get discretization step from dynamics
  ns = 6
  ni = 2
  TT = int(tf/dt) # discrete-time samples

  xx_ref = np.zeros((ns, TT))
  uu_ref = np.zeros((ni, TT))
  ## Insert funtion to TEST
  xx_ref, uu_ref = Skidpad_Curve(ns, ni, TT, dt)

  tt_hor = np.linspace(0,tf,TT)

  fig, axs = plt.subplots(ns+ni, 1, sharex='all')

  axs[0].plot(tt_hor, xx_ref[0,:], 'g--', linewidth=2)
  axs[0].grid()
  axs[0].set_ylabel('$x_0$')

  axs[1].plot(tt_hor, xx_ref[1,:], 'g--', linewidth=2)
  axs[1].grid()
  axs[1].set_ylabel('$x_1$')

  axs[2].plot(tt_hor, xx_ref[2,:], 'g--', linewidth=2)
  axs[2].grid()
  axs[2].set_ylabel('$x_2$')

  axs[3].plot(tt_hor, xx_ref[3,:], 'g--', linewidth=2)
  axs[3].grid()
  axs[3].set_ylabel('$x_3$')

  axs[4].plot(tt_hor, xx_ref[4,:], 'g--', linewidth=2)
  axs[4].grid()
  axs[4].set_ylabel('$x_4$')

  axs[5].plot(tt_hor, xx_ref[5,:], 'g--', linewidth=2)
  axs[5].grid()
  axs[5].set_ylabel('$x_5$')


  axs[6].plot(tt_hor, uu_ref[0,:], 'r--', linewidth=2)
  axs[6].grid()
  axs[6].set_ylabel('$u_0$')

  axs[7].plot(tt_hor, uu_ref[1,:], 'r--', linewidth=2)
  axs[7].grid()
  axs[7].set_ylabel('$u_1$')
    
  # plt.show()

  plt.figure('X, Y')
  plt.plot(xx_ref[0,:],xx_ref[1,:])
  plt.grid()
  plt.show()


# np.deg2rad(ref_deg_T)

