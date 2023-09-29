import numpy as np
import control as ctrl


def lti_LQR(AA, BB, QQ, RR, SS, QQT, TT, qq, rr, ns, ni):

  """
	LQR for LTI system with fixed cost	
	
  Args
    - AA (ns x ns) matrix
    - BB (ns x ni) matrix
    - QQ (ns x ns), RR (mm x mm) stage cost
    - QQT (nn x nn) terminal cost
    - TT time horizon
  
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
    - sigma (ni x 1 x TT)

  """

  #inizializzazione matrici e vettori
  PP = np.zeros((ns,ns,TT)) #(6,6)
  KK = np.zeros((ni,ns,TT)) #(2,6)
  sigma = np.zeros((ni,1,TT)) #vector for deltau #(2,1)
  pp = np.zeros((ns,1,TT))
  
  #inizializzazione PPT e ppT
  PP[:,:,-1] = QQT
  pp[:,:,-1] = qq[:,:,-1]
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    # Per rendere più leggibile
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]
    PPtp = PP[:,:,tt+1]
    pptp = pp[:,:,tt+1]
    qqt = qq[:,:,tt]
    rrt = rr[:,:,tt]

    KK[:,:,tt] = -np.linalg.inv((RRt+ BBt.T@PPtp@BBt))@(SSt+ BBt.T@PPtp@AAt)

    sigma[:,:,tt] = -np.linalg.inv((RRt+ BBt.T@PPtp@BBt))@(rrt+BBt.T@pptp)  #Il termine noto della dinamica è nullo 'cct'

    pp[:,:,tt] = qqt + AAt.T@pptp - KK[:,:,tt].T@(RRt+ BBt.T@PPtp@BBt)@sigma[:,:,tt]  #Il termine noto della dinamica è nullo 'cct'
    
    PP[:,:,tt] = QQt + AAt.T@PPtp@AAt - KK[:,:,tt].T@(RRt+ BBt.T@PPtp@BBt)@KK[:,:,tt]

  return KK, PP, sigma


def ltv_LQP(AA, BB, QQ, RR, QQf, TT, ns, ni):

  """
	LQR for LTV system with (time-varying) cost	
	
  Args
    - AA (nn x nn (x TT)) matrix
    - BB (nn x mm (x TT)) matrix
    - QQ (nn x nn (x TT)), RR (mm x mm (x TT)) stage cost
    - QQf (nn x nn) terminal cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """
	
  # try:
  #   # check if matrix is (.. x .. x TT) - 3 dimensional array 
  #   ns, lA = AA.shape[1:]
  # except:
  #   # if not 3 dimensional array, make it (.. x .. x 1)
  #   AA = AA[:,:,None]
  #   ns, lA = AA.shape[1:]

  # try:  
  #   nu, lB = BB.shape[1:]
  # except:
  #   BB = BB[:,:,None]
  #   ni, lB = BB.shape[1:]

  # try:
  #     nQ, lQ = QQ.shape[1:]
  # except:
  #     QQ = QQ[:,:,None]
  #     nQ, lQ = QQ.shape[1:]

  # try:
  #     nR, lR = RR.shape[1:]
  # except:
  #     RR = RR[:,:,None]
  #     nR, lR = RR.shape[1:]

  # # Check dimensions consistency -- safety
  # if nQ != ns:
  #   print("Matrix Q does not match number of states")
  #   exit()
  # if nR != ni:
  #   print("Matrix R does not match number of inputs")
  #   exit()


  # if lA < TT:
  #     AA = AA.repeat(TT, axis=2)
  # if lB < TT:
  #     BB = BB.repeat(TT, axis=2)
  # if lQ < TT:
  #     QQ = QQ.repeat(TT, axis=2)
  # if lR < TT:
  #     RR = RR.repeat(TT, axis=2)

  PP = np.zeros((ns,ns,TT))
  KK = np.zeros((ni,ns,TT))
  
  PP[:,:,-1] = QQf
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    PPtp = PP[:,:,tt+1]
    
    PP[:,:,tt] = QQt + AAt.T@PPtp@AAt - \
        + (AAt.T@PPtp@BBt)@np.linalg.inv((RRt + BBt.T@PPtp@BBt))@(BBt.T@PPtp@AAt)
  
  # Evaluate KK
  
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    RRt = RR[:,:,tt]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    PPtp = PP[:,:,tt+1]
    
    KK[:,:,tt] = -np.linalg.inv(RRt + BBt.T@PPtp@BBt)@(BBt.T@PPtp@AAt)

  return KK, PP
    