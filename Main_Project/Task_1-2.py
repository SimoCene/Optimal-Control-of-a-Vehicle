# Task-1 & Task-2
# Exploit the Newton’s algorithm for optimal control to compute the optimal trajectory.
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
import solver_LQ as lqp #import lqr solver
# Allow Ctrl-C to work despite plotting
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

plt.rcParams["figure.figsize"] = (10,8)
plt.rcParams.update({'font.size': 22})

np.set_printoptions(suppress=True,  precision=3)  # per printare solo in modalità float (e non esponenziale)



#####################################################################################################################
############################################ - -       TO_DO        - - #############################################
#####################################################################################################################
## Choose the reference curve between 'Step' & 'Skidpad' & 'Sigmoid'
Curve = 'Skidpad'

print_armijo = False
visu_armijo = False
Save_output_opt = False




#####################################################################################################################
############################################ - -       PARAMETERS        - - ########################################
#####################################################################################################################
max_iters = int(2e2)
term_cond = 1e-2 # Terminal confition, value of 'descent'

stepsize_0 = 1
Do_Armijo = True
cc = 0.5  # ARMIJO PARAMETERS
beta = 0.7  # beta = 0.5
armijo_maxiters = 20 # number of Armijo iterations
visu_armijo_sampling = 10  #Definisce la granulatià del grafico, della cost function


#######################################
# Trajectory parameters
#######################################
if Curve == 'Step'  or  Curve == 'Sigmoid':
  Velocity = 1 #[m/s]
  Step_value = 1 #[m] 
  tf = 8*int(Step_value/Velocity) # final time in [seconds], il coeficcente è stato ottenuto a livello sperimentale. (Coeficcente a 32 ottimale ma 16 iter)
  print('Final time =', tf, 's\twith an imposed velocity of ', Velocity, 'm/s\t and step of ', Step_value, 'm')
elif Curve == 'Skidpad':
  Velocity = 3 #[m/s]
  tf = int((2*2*np.pi*9.125)/Velocity)  # final time in seconds
  print('Final time =', tf, 's\twith an imposed velocity of ', Velocity, 'm/s')
  ### If you want impose the final time, uncomment
  # tf = 40 # final time in seconds
  # print('Velocity =', (2*2*np.pi*9.125)/tf, 'm/s\twith an imposed final time of ', tf, 's') #Where radius is 9.125
  ###

dt = 1e-2 # discretization stepsize - Forward Euler
ns = 6  # Number state
ni = 2  # Number input

TT = int(tf/dt) # discrete-time samples


######################################
# Arrays to store data
######################################
xx = np.zeros((ns, TT, max_iters))   # state seq.
uu = np.zeros((ni, TT, max_iters))   # input seq.

lmbd = np.zeros((ns, TT, max_iters)) # lambdas - costate seq.
deltau = np.zeros((ni,TT, max_iters)) # Du - descent direction
gradient = np.zeros((ni,TT, max_iters))
JJ = np.zeros(max_iters)      # collect cost
descent = np.zeros(max_iters) # collect descent direction, the quadratic of Du


if Curve == 'Step'  or  Curve == 'Sigmoid':
  ######################################
  # Reference curve
  ######################################
  if Curve == 'Step':
    xx_ref, uu_ref = Ref.Step_Curve(ns, ni, TT, dt, Velocity, Step_value)
  elif Curve == 'Sigmoid':
    xx_ref, uu_ref = Ref.Sigmoid_Curve(ns, ni, TT, dt, 0, 1, 0, 1,15)
  x0 = xx_ref[:,0]  # Shape = (6,)

  #######################################
  # Cost function parameters - STEP
  #######################################
  QQt = np.eye(ns)  # Pesi degli stati
  QQt[0,0] = 1e5        # X   [m]
  QQt[1,1] = 1e5        # Y   [m]
  QQt[2,2] = 1e2  # Phi [Rad] --> Dimensionalmente piccoli (1e-2)^2
  QQt[3,3] = 1          # Vx  [m/s] --> Dimensionalmente scalati per il dt
  QQt[4,4] = 1          # Vy  [m/s] --> Dimensionalmente scalati per il dt
  QQt[5,5] = 1          # Phi_dot [Rad/s]

  RRt = np.eye(ni)  # Pesi degli ingressi
  RRt[0,0] = 1e4  # Delta [Rad] --> Dimensionalmente piccoli (1e-2)^2
  RRt[1,1] = 1e-3   # Fx    [Newton]  --> Dimensionalmente grande (1e1)^2

  QQT = np.eye(ns)  # Pesi degli stati all'istante finale
  QQT[0,0] = 1e3        # X   [m]
  QQT[1,1] = 1e4        # Y   [m]
  QQT[2,2] = 1e4  # Phi [Rad] --> Dimensionalmente piccoli (1e-2)^2
  QQT[3,3] = 1           # Vx  [m/s] --> Dimensionalmente scalati per il dt
  QQT[4,4] = 1           # Vy  [m/s] --> Dimensionalmente scalati per il dt
  QQT[5,5] = 1           # Phi_dot [Rad/s]

  ######################################
  # Initial guess trajectory
  ######################################
  xx[:,:,0], uu[:,:,0] = Init.Line_Trj(ns, ni, TT, dt, Velocity)



elif Curve == 'Skidpad':
  ######################################
  # Reference curve
  ######################################
  xx_ref, uu_ref = Ref.Skidpad_Curve(ns, ni, TT, dt)
  x0 = xx_ref[:,0]  # Shape = (6,)

  #######################################
  # Cost function parameters - SKIDPAD
  #######################################
  QQt = np.eye(ns)  # Pesi degli stati
  QQt[0,0] = 1e3        # X   [m]
  QQt[1,1] = 1e3        # Y   [m]
  QQt[2,2] = 1          # Phi [Rad]
  QQt[3,3] = 1          # Vx  [m/s]
  QQt[4,4] = 1          # Vy  [m/s]
  QQt[5,5] = 1          # Phi_dot [Rad/s]

  RRt = np.eye(ni)  # Pesi degli ingressi, per compensare l'unità di misura e renderli unitari
  # Rad, valore medio    -> 0.04 = 4*1e-2 -> (4*1e-2)^2 = 4*1e-4    ==> 4*1e-4 * 0.25*1e4  == [1]     {moltiplicatore 2.5*1e3 }
  # Newton, valore medio  -> 250 = 25*1e1 -> (25*1e1)^2 = 25*1e2    ==> 25*1e2 * 0.04*1e-2 == [1]     {moltiplicatore 4*1e-4 }
  RRt[0,0] = 2.5*1e3    # Delta [Rad]
  RRt[1,1] = 4*1e-4     # Fx    [Newton]

  QQT = np.eye(ns)  # Pesi degli stati all'istante finale
  QQT[0,0] = 1e4        # X   [m]
  QQT[1,1] = 1e4        # Y   [m]
  QQT[2,2] = 1e4        # Phi [Rad]
  QQT[3,3] = 1          # Vx  [m/s]
  QQT[4,4] = 1          # Vy  [m/s]
  QQT[5,5] = 1          # Phi_dot [Rad/s]

  ######################################
  # Initial guess Trajectory
  ######################################
  xx[:,:,0], uu[:,:,0] = Init.PID_Trj_2(ns,ni,TT,dt,xx_ref)





#####################################################################################################################
############################################ - -       MAIN        - - ##############################################
#####################################################################################################################
print('-*-*-*-*-*-')
kk = 0

for kk in range(max_iters-1):
  # Definizione delle matrici, azzerate per ogni nuova iterazione. Così non sono in 4D (senza kk)
  AA=np.zeros((ns,ns,TT)) #(6,6)
  BB=np.zeros((ns,ni,TT)) #(6,2)
  QQ=np.zeros((ns,ns,TT)) #(6,6)
  RR=np.zeros((ni,ni,TT)) #(2,2)
  SS=np.zeros((ni,ns,TT)) #(2,6)
  qq=np.zeros((ns,1,TT))  #(6,1)
  rr=np.zeros((ni,1,TT))  #(2,1)
  deltax=np.zeros((ns,1,TT)) #(6,1) #deltax iniziale sempre = 0

  KK = np.zeros((ni,ns,TT)) #(2,6)
  PP = np.zeros((ns,ns,TT)) #(6,6)
  sigma = np.zeros((ni,1,TT)) #(2,1) #vector for deltau


  ## ##################################################################
  ## calculate Cost function
  ## ##################################################################
  JJ[kk] = 0
  for tt in range(TT-1):
    temp_cost = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[0]
    JJ[kk] += temp_cost
  
  temp_cost = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[0]  ## TT-1 == -1
  JJ[kk] += temp_cost


  ## ##################################################################
  ## Inizializaione delle matrici regolarizzate
  ## ##################################################################
  for tt in range(TT):
    aa, bb = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[1:] # aa -> (6,1) , bb -> (2,1)
    fx, fu = dyn.dynamics(xx[:,tt,kk], uu[:,tt,kk], dt)[1:] # fx -> (6,6) , fu -> (6,2)

    AA[:,:,tt] = fx #fx is a Jacobian --> AA still a Jacobian (6,6)
    BB[:,:,tt] = fu #fu is a Jacobian --> BB still a Jacobian (6,2)
    QQ[:,:,tt] = QQt  # (6,6) # Newton regolarizzato, solo la derivata seconda in xx della Cost function
    RR[:,:,tt] = RRt  # (2,2) # Newton regolarizzato, solo la derivata seconda in uu della Cost function
    # SS[:,:,tt] =    # Newton regolarizzato, solo la derivata seconda in xx & uu della Cost function. In questo caso 0
    qq[:,:,tt]=aa # (6,1)
    rr[:,:,tt]=bb # (2,1)


  ## ##################################################################
  ## Solve Riccati equation, backward in time
  ## ##################################################################
  KK, PP, sigma = lqp.lti_LQR(AA,BB,QQ,RR,SS,QQT,TT,qq,rr, ns, ni)  # KK -> (2,6), PP -> (6,6), sigma -> (2,1)

  ## ##################################################################
  ## Solve the Delta_U with Newton, forward in time
  ## ##################################################################
  for tt in range(TT-1):
    deltau[:,tt,kk] = (KK[:,:,tt]@deltax[:,:,tt] + sigma[:,:,tt]).squeeze() #(2,) = [(2,6)*(6,1) + (2,1)].squeeze
    deltax[:,:,tt+1] = AA[:,:,tt]@deltax[:,:,tt] + BB[:,:,tt]@deltau[:,tt,kk,None] #(6,1) = (6,6)*(6,1) + (6,2)*(2,'1')


  ## ##################################################################
  ## Gradient methods for descent direction
  ## ##################################################################
  lmbd_temp = cst.termcost(xx[:,TT-1,kk], xx_ref[:,TT-1], QQT)[1]  ## TT-1 == -1 -> (6,1)
  lmbd[:,TT-1,kk] = lmbd_temp.squeeze() # -> (6,)  #Calculate lamda for costate equation

  for tt in reversed(range(TT-1)):  # integration backward in time
    # Remember that now you need the Gradient of the dynamics
    lmbd_temp = AA[:,:,tt].T@lmbd[:,tt+1,kk][:,None] + qq[:,:,tt]   # costate equation  # Return (6,1) = (6,6)*(6,1) + (6,1)
    grad_temp = - BB[:,:,tt].T@lmbd[:,tt+1,kk][:,None] - rr[:,:,tt] # Return (2,1) = (2,6)*(6,1) + (2,1)

    lmbd[:,tt,kk] = lmbd_temp.squeeze()     ## lmbd_temp.squeeze() -> (6,)
    gradient[:,tt,kk] = grad_temp.squeeze() ## grad_temp.squeeze() -> (2,)


  ## ##################################################################
  ## Calculate the descent direction for Armijo. Descent += Gradeint@Delta_u
  ## ##################################################################
  for tt in range(TT-1):
    #descent[kk] += deltau[:,tt,kk].T@deltau[:,tt,kk] # Scalare = (,2)*(2,)
    descent[kk] = descent[kk] + gradient[:,tt,kk].T@deltau[:,tt,kk] # Scalare = (,2)*(2,)


  ## ##################################################################
  ## Stepsize selection - ARMIJO or COSTANT
  ## ##################################################################
  stepsize = stepsize_0
  if Do_Armijo:
    stepsizes = []  # list of stepsizes
    costs_armijo = []

    Iter_Armijo = 0
    
    for Iter_Armijo in range(armijo_maxiters):

      # temp solution update
      xx_temp = np.zeros((ns,TT))
      uu_temp = np.zeros((ni,TT))
      xx_temp[:,0] = x0

      for tt in range(TT-1):
        uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt,kk]
        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt], dt)[0]

      # temp cost calculation
      JJ_temp = 0
      for tt in range(TT-1):
        temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[0]
        JJ_temp += temp_cost

      temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1], QQT)[0]
      JJ_temp += temp_cost

      stepsizes.append(stepsize)      # save the stepsize
      costs_armijo.append(JJ_temp)    # save the cost associated to the stepsize

      if JJ_temp > JJ[kk] - cc*stepsize*descent[kk]:
          # update the stepsize
          stepsize = beta*stepsize
      
      else:
          # print('Armijo stepsize = {}\t Iter_Armijo= {}'.format(stepsize,Iter_Armijo))
          break

    if print_armijo:
      print('Armijo stepsize = {}\t Iter_Armijo= {}'.format(stepsize,Iter_Armijo))

  ## ##################################################################
  ## Armijo PLOT
  ## ##################################################################

  if visu_armijo:

    steps = np.linspace(0,stepsize_0,int(visu_armijo_sampling))  #Definisce la granulatià del grafico, della cost function
    costs = np.zeros(len(steps))

    for ii in range(len(steps)):

      step = steps[ii]

      # temp solution update

      xx_temp = np.zeros((ns,TT))
      uu_temp = np.zeros((ni,TT))

      xx_temp[:,0] = x0

      for tt in range(TT-1):
        uu_temp[:,tt] = uu[:,tt,kk] + step*deltau[:,tt,kk]
        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt],dt)[0]

      # temp cost calculation
      JJ_temp = 0

      for tt in range(TT-1):
        temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], QQt, RRt)[0]
        JJ_temp += temp_cost

      temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1], QQT)[0]
      JJ_temp += temp_cost

      costs[ii] = JJ_temp


    plt.figure('Armijo PLOT')
    plt.clf()

    plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
    plt.plot(steps, JJ[kk] - descent[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    plt.plot(steps, JJ[kk] - cc*descent[kk]*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

    plt.scatter(stepsizes, costs_armijo, marker='*', label='Selection of stepsizes') # plot the tested stepsize

    plt.grid()
    plt.xlabel('stepsize')
    plt.ylabel('$J(\\mathbf{u}^k,{stepsize})$')
    plt.legend()
    plt.draw()
    
    plt.title('Stepsize selection, number of iteration: {}' .format(Iter_Armijo+1))
    plt.show(block=False)
    plt.pause(1)


  ## ##################################################################
  ## Update the current solution
  ## ##################################################################
  xx_temp = np.zeros((ns,TT))
  uu_temp = np.zeros((ni,TT))
  xx_temp[:,0] = x0

  for tt in range(TT-1):
    uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt,kk]
    xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt], dt)[0]

  xx[:,:,kk+1] = xx_temp
  uu[:,:,kk+1] = uu_temp


  ## ##################################################################
  ## Termination condition
  ## ##################################################################
  print('Iter = {}\t Descent = {}\t Cost = {}'.format(kk,descent[kk],JJ[kk]))

  if descent[kk] <= term_cond:

    max_iters = kk

    break

xx_star = xx[:,:,max_iters-1]
uu_star = uu[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2] # for plot, because you don't have the last value
# Save the inital trajectory for plot
xx_init = xx[:,:,0]
uu_init = uu[:,:,0]
uu_init[:,-1] = uu_init[:,-2] # for plot, because you don't have the last value





#####################################################################################################################
############################################ - -       Save Data        - - ##############################################
#####################################################################################################################
if Save_output_opt:
  np.savetxt('xx_opt_{}_Iter{}.txt'.format(Curve, max_iters),xx_star,delimiter=',')
  np.savetxt('uu_opt_{}_Iter{}.txt'.format(Curve, max_iters),uu_star,delimiter=',')




#####################################################################################################################
############################################ - -       PLOTS        - - ##############################################
#####################################################################################################################
# cost and descent
plt.figure('descent direction')
plt.plot(np.arange(max_iters), descent[:max_iters])
plt.xlabel('$k$')
plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
plt.yscale('log')
plt.grid()
plt.title('Descent direction')
plt.show(block=False)

plt.figure('Cost')
plt.plot(np.arange(max_iters), JJ[:max_iters])
plt.xlabel('$k$')
plt.ylabel('$J(\\mathbf{u}^k)$')
plt.yscale('log')
plt.grid()
plt.title('Cost function')
plt.show(block=False)

# optimal trajectory
tt_hor = np.linspace(0,tf,TT)
# fig_X_1 = plt.figure('X_0-1-2')
fig_X_1, axs = plt.subplots(int(ns/2), 1, sharex='all')
for ii in range(0,int(ns/2)):
  axs[ii].plot(tt_hor, xx_ref[ii,:], 'g--', linewidth=2, label='$Ref$')
  axs[ii].plot(tt_hor, xx_init[ii,:], 'k', linewidth=2, label='$Init$')
  axs[ii].plot(tt_hor, xx_star[ii,:], 'r', linewidth=2, label='$Opt$')
  axs[ii].grid()
  axs[ii].set_ylabel('$x_{}$'.format(ii))
axs[ii].set_xlabel('time')
fig_X_1.legend(['$Ref$', '$Init$', '$Opt$'], loc='lower right', ncol = 3, fontsize = 10)
fig_X_1.suptitle('Evolution of State [0,1,2]')
plt.show(block=False)
# fig_X_2 = plt.figure('X_3-4-5')
fig_X_2, axs = plt.subplots(int(ns/2), 1, sharex='all')
for ii in range(int(ns/2),ns):
  axs[ii-int(ns/2)].plot(tt_hor, xx_ref[ii,:], 'g--', linewidth=2, label='$Ref$')
  axs[ii-int(ns/2)].plot(tt_hor, xx_init[ii,:], 'k', linewidth=2, label='$Init$')
  axs[ii-int(ns/2)].plot(tt_hor, xx_star[ii,:], 'r', linewidth=2, label='$Opt$')
  axs[ii-int(ns/2)].grid()
  axs[ii-int(ns/2)].set_ylabel('$x_{}$'.format(ii))
axs[ii-int(ns/2)].set_xlabel('time')
fig_X_2.legend(['$Ref$', '$Init$', '$Opt$'], loc='lower right', ncol = 3, fontsize = 10)
fig_X_2.suptitle('Evolution of State [3,4,5]')
plt.show(block=False)

# fig_U = plt.figure('U')  
fig_U, axs = plt.subplots(ni, 1, sharex='all')
for ii in range(ni):
  axs[ii].plot(tt_hor, uu_ref[ii,:], 'g--', linewidth=2, label='$Ref$')
  axs[ii].plot(tt_hor, uu_init[ii,:], 'k', linewidth=2, label='$Init$')
  axs[ii].plot(tt_hor, uu_star[ii,:], 'r', linewidth=2, label='$Opt$')
  axs[ii].grid()
  axs[ii].set_ylabel('$u_{}$'.format(ii))
axs[ii].set_xlabel('time')
fig_U.legend(['$Ref$', '$Init$', '$Opt$'], loc='lower right', ncol = 3, fontsize = 10)
fig_U.suptitle('Evolution of Input')
plt.show(block=False)

# Evolution on the plane
fig_plane = plt.figure('plane')
for kk in range(1,max_iters-1-1,2):
  plt.plot(xx[0,:,kk], xx[1,:,kk], linewidth=2, label='iter_{}'.format(kk))
plt.plot(xx[0,:,0], xx[1,:,0], 'k', linewidth=2, label='$Init$')
plt.plot(xx_ref[0,:], xx_ref[1,:], 'g--', linewidth=2.5, label='$Ref$')
plt.plot(xx_star[0,:], xx_star[1,:], 'r:', linewidth=2.5, label='$Opt$')
plt.grid()
plt.ylabel('$Y$')
plt.xlabel('$X$')
fig_plane.legend(loc='lower right', ncol = 2, fontsize = 10)
plt.title('Evolution on the plane')
if Curve == 'Skidpad': plt.axis('equal')
plt.show(block=False)

# To plot all togheter
plt.show()