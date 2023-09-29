# Implementation of cost funciton
# Gradient method for Optimal Control
# Cost functions
# Group 19
# Bologna, 24/01/2023
#

import numpy as np


def stagecost(xx,uu, xx_ref, uu_ref, QQt, RRt):
  """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args
      - xx \in \R^ns state at time t
      - xx_ref \in \R^ns state reference at time t

      - uu \in \R^ni input at time t
      - uu_ref \in \R^ni input reference at time t


    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """
  # Per convertire da (6,) to (6,1)
  xx = xx[:,None]
  uu = uu[:,None]

  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref) # Return scalar (1,1) = (1,6)@(6,6)@(6,1)

  lx = QQt@(xx - xx_ref) # Return (6,1) = (6,6)@(6,1)
  lu = RRt@(uu - uu_ref)

  return ll, lx, lu

def termcost(xx,xx_ref, QQT):
  """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """
  
  # Per compensare il problema di (6,) to (6,1)
  xx = xx[:,None]
  xx_ref = xx_ref[:,None]

  llT = 0.5*(xx - xx_ref).T@QQT@(xx - xx_ref) # Return scalar (1,6)@(6,6)@(6,1)

  lTx = QQT@(xx - xx_ref) # Return (6,1) = (6,6)@(6,1)

  return llT, lTx




#####################################################################################################################
############################################ - -       Barrier function        - - ##################################
#####################################################################################################################
def stagecost_barrier_function(xx,uu, xx_ref, uu_ref, QQt, RRt, uu_limt, epsilon_for_log):
  """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args
      - xx \in \R^ns state at time t
      - xx_ref \in \R^ns state reference at time t

      - uu \in \R^ni input at time t
      - uu_ref \in \R^ni input reference at time t


    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """
  # Per convertire da (6,) to (6,1)
  xx = xx[:,None]
  uu = uu[:,None]

  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  # Limit on the inputs
  ni = np.shape(uu)[0]
  uu_limt = uu_limt[:,None] #(2,1)
  # Set the argument of the barrier function
  gg_u = np.zeros(ni)
  gg_u = gg_u[:,None] #(2,1)
  gg_u[0] = (((uu[0] / uu_limt[0])**2) - 1)
  gg_u[1] = (((uu[1] / uu_limt[1])**2) - 1)
  # Se il valore di super il limite l'argomento diventa negativo. 
  # Il log (-x) non è definito, quindi se diventa negativo viene bloccato a circa 0 (1e-15)
  for ii in range(ni):
    if gg_u[ii] > -(1e-20):
      gg_u[ii] = -(1e-20)


  ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)  # Return scalar (1,1) = (1,6)@(6,6)@(6,1)
  ll = ll + epsilon_for_log*(-np.log2(-gg_u[0])) + epsilon_for_log*(-np.log2(-gg_u[1])) #Somma della barrier function
  # print('Epsilon = {}\t Argomento di log = {}\t Cost_barrier = {}'.format(epsilon_for_log,-gg_u[1],epsilon_for_log * (-np.log2(-gg_u[1]))))


  lx = QQt@(xx - xx_ref) # Return (6,1) = (6,6)@(6,1)
  lu = RRt@(uu - uu_ref)

  return ll, lx, lu