# Task-3
# Trajectory tracking to define the optimal feedback controller
# Main
# Group 19
# Bologna, 26/01/2023
#

import numpy as np
import matplotlib.pyplot as plt
import Dynamics_Numpy as dyn  # import car dynamics
import cost as cst  # import cost functions
import Reference_curve as Ref   # import reference curve
import Initial_Trajectory as Init # import initial guess trajectory
import solver_LQ as lqp #import lqp solver
# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
from matplotlib.animation import FuncAnimation  #For the animation

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

np.set_printoptions(suppress=True,  precision=3)  # per printare solo in modalità float (e non esponenziale)


#####################################################################################################################
############################################ - -       TO_DO        - - ########################################
#####################################################################################################################
### Choose the optimal trajectory between 'Step' & 'Skidpad'
Trajectory = 'Skidpad'
###

# Load the optimal trajectory obtained with 'Task_1-2'
if Trajectory == 'Step':
  xx_opt = np.loadtxt('xx_opt_Step.txt', delimiter=',')
  uu_opt = np.loadtxt('uu_opt_Step.txt', delimiter=',')
elif Trajectory == 'Skidpad':
  xx_opt = np.loadtxt('xx_opt_Skidpad.txt', delimiter=',')
  uu_opt = np.loadtxt('uu_opt_Skidpad.txt', delimiter=',')




#####################################################################################################################
############################################ - -       PARAMETERS        - - ########################################
#####################################################################################################################
TT = np.shape(xx_opt)[1]  # discrete-time samples
dt = 1e-2 # discretization stepsize - Forward Euler
ns = 6  # Number state
ni = 2  # Number input
tf = int(TT*dt) # final time


if Trajectory == 'Step':
  #######################################
  # Disturbances on initial state
  #######################################
  dis_x0 = np.zeros(ns)
  dis_x0[0] = 0
  dis_x0[1] = 0.5
  dis_x0[2] = 0
  dis_x0[3] = 0
  dis_x0[4] = 0
  dis_x0[5] = 0
  # Disturbances at time
  Insert_dis_time = True
  dis_time = int(TT*(2/4))
  dis_tt = np.zeros(ns)
  dis_tt[0] = -0.5
  dis_tt[1] = -0.5
  dis_tt[2] = 0
  dis_tt[3] = 0
  dis_tt[4] = 0
  dis_tt[5] = 0

  #######################################
  # Weights for LQP
  #######################################
  QQt = np.eye(ns)  # Pesi degli stati
  QQt[0,0] = 1e2     # X   [m]
  QQt[1,1] = 1e3     # Y   [m]
  QQt[2,2] = 1      # Phi [Rad] --> Dimensionalmente piccoli (1e-2)^2
  QQt[3,3] = 1      # Vx  [m/s] --> Dimensionalmente scalati per il dt
  QQt[4,4] = 1      # Vy  [m/s] --> Dimensionalmente scalati per il dt
  QQt[5,5] = 1      # Phi_dot [Rad/s]

  RRt = np.eye(ni)  # Pesi degli ingressi
  RRt[0,0] = 1e4    # Delta [Rad] --> Dimensionalmente piccoli (1e-2)^2
  RRt[1,1] = 1e-3   # Fx    [Newton]  --> Dimensionalmente grande (1e1)^2

  QQT = np.eye(ns)  # Pesi degli stati all'istante finale
  QQT[0,0] = 1e1      # X   [m]
  QQT[1,1] = 1e1      # Y   [m]
  QQT[2,2] = 1      # Phi [Rad] --> Dimensionalmente piccoli (1e-2)^2
  QQT[3,3] = 1      # Vx  [m/s] --> Dimensionalmente scalati per il dt
  QQT[4,4] = 1      # Vy  [m/s] --> Dimensionalmente scalati per il dt
  QQT[5,5] = 1      # Phi_dot [Rad/s]



elif Trajectory == 'Skidpad':
  #######################################
  # Disturbances on initial state
  #######################################
  dis_x0 = np.zeros(ns)
  dis_x0[0] = 5
  dis_x0[1] = -1
  dis_x0[2] = 0
  dis_x0[3] = 0
  dis_x0[4] = 0
  dis_x0[5] = 0
  # Disturbances at time
  Insert_dis_time = True
  dis_time = int(TT*(3/4))
  dis_tt = np.zeros(ns)
  dis_tt[0] = 1
  dis_tt[1] = -1
  dis_tt[2] = 0
  dis_tt[3] = 0
  dis_tt[4] = 0
  dis_tt[5] = 0

  #######################################
  # Weights for LQP
  #######################################
  QQt = np.eye(ns)  # Pesi degli stati
  QQt[0,0] = 100      # X   [m]
  QQt[1,1] = 100      # Y   [m]
  QQt[2,2] = 1      # Phi [Rad] --> Dimensionalmente piccoli (1e-2)^2
  QQt[3,3] = 1      # Vx  [m/s] --> Dimensionalmente scalati per il dt
  QQt[4,4] = 1      # Vy  [m/s] --> Dimensionalmente scalati per il dt
  QQt[5,5] = 1      # Phi_dot [Rad/s]

  RRt = np.eye(ni)  # Pesi degli ingressi
  RRt[0,0] = 1e4    # Delta [Rad] --> Dimensionalmente piccoli (1e-2)^2
  RRt[1,1] = 1e-3   # Fx    [Newton]  --> Dimensionalmente grande (1e1)^2

  QQT = np.eye(ns)  # Pesi degli stati all'istante finale
  QQT[0,0] = 10      # X   [m]
  QQT[1,1] = 10      # Y   [m]
  QQT[2,2] = 1      # Phi [Rad] --> Dimensionalmente piccoli (1e-2)^2
  QQT[3,3] = 1      # Vx  [m/s] --> Dimensionalmente scalati per il dt
  QQT[4,4] = 1      # Vy  [m/s] --> Dimensionalmente scalati per il dt
  QQT[5,5] = 1      # Phi_dot [Rad/s]


######################################
# Arrays to store data
######################################
xx_reg = np.zeros((ns,TT))
uu_reg = np.zeros((ni,TT))

AA=np.zeros((ns,ns,TT)) #(6,6)
BB=np.zeros((ns,ni,TT)) #(6,2)
QQ=np.zeros((ns,ns,TT)) #(6,6)
RR=np.zeros((ni,ni,TT)) #(2,2)
KK = np.zeros((ni,ns,TT)) #(2,6)
PP = np.zeros((ns,ns,TT)) #(6,6)




#####################################################################################################################
############################################ - -       MAIN        - - ##############################################
#####################################################################################################################
print('-*-*-*-*-*-')

for tt in range(TT):
  fx, fu = dyn.dynamics(xx_opt[:,tt], uu_opt[:,tt], dt)[1:] # fx -> (6,6) , fu -> (6,2)

  AA[:,:,tt] = fx #fx is a Jacobian --> AA still a Jacobian (6,6)
  BB[:,:,tt] = fu #fu is a Jacobian --> BB still a Jacobian (6,2)
  QQ[:,:,tt] = QQt  # (6,6) # Copio Le matrici dei costi perchè sono time variant
  RR[:,:,tt] = RRt  # (2,2) # Copio Le matrici dei costi perchè sono time variant

KK,PP = lqp.ltv_LQP(AA,BB,QQ,RR,QQT,TT,ns,ni) #Linear Quadratic Problem to define the optimal feedback controller (KK gain)

######################################
# Insert initial disturbance
######################################
xx_reg[0,0] = xx_opt[0,0] + dis_x0[0]
xx_reg[1,0] = xx_opt[1,0] + dis_x0[1]
xx_reg[2,0] = xx_opt[2,0] + dis_x0[2]
xx_reg[3,0] = xx_opt[3,0] + dis_x0[3]
xx_reg[4,0] = xx_opt[4,0] + dis_x0[4]
xx_reg[5,0] = xx_opt[5,0] + dis_x0[5]

######################################
# Control closed-loop - optimal feedback controller
######################################
for tt in range(TT-1):
  if tt == dis_time  and  Insert_dis_time:  # Insert disturbance
    xx_reg[0,dis_time] = xx_opt[0,dis_time] + dis_tt[0]
    xx_reg[1,dis_time] = xx_opt[1,dis_time] + dis_tt[1]
    xx_reg[2,dis_time] = xx_opt[2,dis_time] + dis_tt[2]
    xx_reg[3,dis_time] = xx_opt[3,dis_time] + dis_tt[3]
    xx_reg[4,dis_time] = xx_opt[4,dis_time] + dis_tt[4]
    xx_reg[5,dis_time] = xx_opt[5,dis_time] + dis_tt[5]
  
  uu_reg[:,tt] = uu_opt[:,tt] + KK[:,:,tt]@(xx_reg[:,tt]-xx_opt[:,tt])
  xx_reg[:,tt+1] = dyn.dynamics(xx_reg[:,tt], uu_reg[:,tt], dt)[0]
# For plot, because you don't have the last value
uu_reg[:,-1] = uu_reg[:,-2]




#####################################################################################################################
############################################ - -       PLOTS        - - #############################################
#####################################################################################################################
# optimal feedback controller
tt_hor = np.linspace(0,tf,TT)
fig_X, axs = plt.subplots(int(ns/2), 1, sharex='all')
for ii in range(0,int(ns/2)):
  axs[ii].plot(tt_hor, xx_opt[ii,:], 'r--', linewidth=2, label='$Opt$')
  axs[ii].plot(tt_hor, xx_reg[ii,:], 'b', linewidth=2.5, label='$Ctr$')
  axs[ii].grid()
  axs[ii].set_ylabel('$x_{}$'.format(ii))
axs[ii].set_xlabel('time')
fig_X.legend(['$Opt$', '$Ctr$'], loc='lower right', ncol = 2, fontsize = 10)
fig_X.suptitle('Evolution of State [0,1,2]')
plt.show(block=False)
fig_X, axs = plt.subplots(int(ns/2), 1, sharex='all')
for ii in range(int(ns/2),ns):
  axs[ii-int(ns/2)].plot(tt_hor, xx_opt[ii,:], 'r--', linewidth=2, label='$Opt$')
  axs[ii-int(ns/2)].plot(tt_hor, xx_reg[ii,:], 'b', linewidth=2.5, label='$Ctr$')
  axs[ii-int(ns/2)].grid()
  axs[ii-int(ns/2)].set_ylabel('$x_{}$'.format(ii))
axs[ii-int(ns/2)].set_xlabel('time')
fig_X.legend(['$Opt$', '$Ctr$'], loc='lower right', ncol = 2, fontsize = 10)
fig_X.suptitle('Evolution of State [3,4,5]')
plt.show(block=False)
  
fig_U, axs = plt.subplots(ni, 1, sharex='all')
for ii in range(ni):
  axs[ii].plot(tt_hor, uu_opt[ii,:], 'r--', linewidth=2, label='$Opt$')
  axs[ii].plot(tt_hor, uu_reg[ii,:], 'b', linewidth=2.5, label='$Ctr$')
  axs[ii].grid()
  axs[ii].set_ylabel('$u_{}$'.format(ii))
axs[ii].set_xlabel('time')
fig_U.legend(['$Opt$', '$Ctr$'], loc='lower right', ncol = 2, fontsize = 10)
fig_U.suptitle('Evolution of Input')
plt.show(block=False)

# Evolution on the plane
fig_plane = plt.figure('plane')
plt.plot(xx_opt[0,:], xx_opt[1,:], 'r--', linewidth=2, label='$Opt$')
plt.plot(xx_reg[0,:], xx_reg[1,:], 'b', linewidth=2.5, label='$Ctr$')
plt.grid()
plt.ylabel('$Y$')
plt.xlabel('$X$')
fig_plane.legend(loc='lower right', ncol = 2, fontsize = 10)
plt.title('Evolution on the plane')
if Trajectory == 'Skidpad': plt.axis('equal')
plt.show(block=False)





#####################################################################################################################
############################################ - -       ANIMATION        - - #########################################
#####################################################################################################################
X_reg = xx_reg[0,:]
Y_reg = xx_reg[1,:]
delta = uu_reg[0,:]
psi   = xx_reg[2,:]
#Import parameter
aa=dyn.aa
bb=dyn.bb

#######################################
# Position of the vaicle
#######################################
# Front position of veicle 
X_front = X_reg + aa*np.cos(psi)
Y_front = Y_reg + aa*np.sin(psi)
# Rear position of veicle 
X_rear = X_reg - bb*np.cos(psi)
Y_rear = Y_reg - bb*np.sin(psi)
# Steering position front wheel
X_t_f = X_front + aa/3*np.cos(delta+psi)
X_t_r = X_front - aa/3*np.cos(delta+psi)
Y_t_f = Y_front + aa/3*np.sin(delta+psi)
Y_t_r = Y_front - aa/3*np.sin(delta+psi)
X_t_f = np.append(X_t_f, X_t_f[-1])
Y_t_f = np.append(Y_t_f, Y_t_f[-1])
X_t_r = np.append(X_t_r, X_t_r[-1])
Y_t_r = np.append(Y_t_r, Y_t_r[-1])
# Steering position back wheel
X_back_f = X_rear + aa/3*np.cos(psi)
X_back_r = X_rear - aa/3*np.cos(psi)
Y_back_f = Y_rear + aa/3*np.sin(psi)
Y_back_r = Y_rear - aa/3*np.sin(psi)
X_back_f = np.append(X_back_f, X_back_f[-1])
Y_back_f = np.append(Y_back_f, Y_back_f[-1])
X_back_r = np.append(X_back_r, X_back_r[-1])
Y_back_r = np.append(Y_back_r, Y_back_r[-1])

#######################################
# Position of the marciapaid
#######################################
rr_Int = 15.25/2
rr_Ext = 21.25/2
rr = (rr_Int+rr_Ext)/2
theta = np.linspace(0, 2*np.pi, 361)
centre_right = rr
centre_left = -rr
alfa = np.arccos(.5*(rr*2)/rr_Ext)

def circle(radius, theta, centre_x): 
    xx = radius*np.cos(theta) + centre_x
    yy = radius*np.sin(theta)
    return list(xx), list(yy)
xx1, yy1 = circle(rr_Ext, np.linspace(0, np.pi - alfa, 181), centre_right)
xx2, yy2 = circle(rr_Ext, np.linspace(alfa, 2*np.pi - alfa, 361), centre_left)
xx3, yy3 = circle(rr_Ext, np.linspace(np.pi + alfa, 2*np.pi, 181), centre_right)

xx_ext = xx1+xx2+xx3
yy_ext = yy1+yy2+yy3

xx_int_left, yy_int_left = circle(rr_Int, theta, centre_left)
xx_int_right, yy_int_right = circle(rr_Int, theta, centre_right)


#######################################
# Creazione animazione
######################################
fig = plt.figure('animation')
ax = fig.add_subplot(1,1,1) #, aspect='equal', autoscale_on=False, xlim=(-22,22), ylim=(-12,12))

## plot statici ##
ax.plot(xx_opt[0,:], xx_opt[1,:], 'r--', linewidth=2, label='$Opt$')
# ax.plot(xx_int_left, yy_int_left, 'k', linewidth=1, label="Marciapaid")
# ax.plot(xx_int_right, yy_int_right, 'k', linewidth=1, label="Marciapaid")
# ax.plot(xx_ext, yy_ext, 'k', linewidth=1, label="Marciapaid")

## plot animati ##
plt.title('Animation')
Chassis, = ax.plot([], [], 'o-c', linewidth=5, label='$Chassis$')
Steering, = ax.plot([], [], '.-m', linewidth=3, label='$SteeringWheel$')
Back_wheel, = ax.plot([], [], '.-r', linewidth=3, label='$BackWheel$')
path, = ax.plot(xx_reg[0,:1], xx_reg[1,:1], 'b', linewidth=2.5, label='$Ctr$')  # Drow the path
# Definisci la funzione di animazione
def update(frame):
  x_points_Chassis  = [X_rear[frame], X_front[frame]]
  y_points_Chassis  = [Y_rear[frame], Y_front[frame]]
  x_points_Steering = [X_t_r[frame], X_t_f[frame]]
  y_points_Steering = [Y_t_r[frame], Y_t_f[frame]]
  x_points_back = [X_back_r[frame], X_back_f[frame]]
  y_points_back = [Y_back_r[frame], Y_back_f[frame]]

  Chassis.set_data(x_points_Chassis, y_points_Chassis)
  Steering.set_data(x_points_Steering, y_points_Steering)
  Back_wheel.set_data(x_points_back, y_points_back)
  path.set_data(xx_reg[0,:frame], xx_reg[1,:frame])
  return Chassis, Steering, Back_wheel, path,
# Funczione animazione
ani = FuncAnimation(fig, update, frames=len(X_reg), interval=2, blit=True, repeat=False)  #Use 'interval' to set the velocity

## Paramtri ##
ax.set_aspect('equal')#, adjustable='box')
ax.grid()
ax.set_yticklabels([])  # no labels
ax.set_xticklabels([])  # no labels
fig.legend(loc='lower center', ncol = 5, fontsize = 10)
## to save animation, uncomment the line below:
## ani.save('offset_piston_motion_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show(block=False)


# To plot all togheter
plt.show()
