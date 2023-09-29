# Dynamics function in symbolic way
# Discrete-time nonlinear dynamics
# Group 19
# Bologna, 17/01/23
#

import sympy as sym

from sympy import sin, cos, symbols
from sympy import pprint


ns = 6
ni = 2


######### - Definition of Symbols - ########
x0    = symbols('x0')
x1    = symbols('x1')
x2    = symbols('x2')
x3    = symbols('x3')
x4    = symbols('x4')
x5    = symbols('x5')

u0    = symbols('u0')
u1    = symbols('u1')

xxp0  = symbols('xxp0')
xxp1  = symbols('xxp1')
xxp2  = symbols('xxp2')
xxp3  = symbols('xxp3')
xxp4  = symbols('xxp4')
xxp5  = symbols('xxp5')

dt    = symbols('dt')

mm    = symbols('mm')
Iz    = symbols('Iz')
aa    = symbols('aa')
bb    = symbols('bb')
mu    = symbols('mu')
gg    = symbols('gg')

Fz_f  = (mm*gg*bb)/(aa+bb)
Fz_r  = (mm*gg*aa)/(aa+bb)

B_f   = u0 - ((x4 + aa*x5)/x3)
B_r   = - ((x4 + bb*x5)/x3)

Fy_f = mu*Fz_f*B_f
Fy_r = mu*Fz_r*B_r


#Forward Euler discretization
xxp0 = x0 + dt * (x3*cos(x2)-x4*sin(x2))
xxp1 = x1 + dt * (x3*sin(x2)+x4*cos(x2))
xxp2 = x2 + dt * (x5)
xxp3 = x3 + dt * ((1/mm)*(u1*cos(u0) - Fy_f*sin(u0)) + x5*x4)
xxp4 = x4 + dt * ((1/mm)*(u1*sin(u0) + Fy_f*cos(u0) + Fy_r) - x5*x3)
xxp5 = x5 + dt * ((1/Iz)*((u1*sin(u0) + Fy_f*cos(u0))*aa - Fy_r*bb))


#################-matrix form-#######################
xxP = sym.zeros(1,ns)
xxP[0]=xxp0
xxP[1]=xxp1
xxP[2]=xxp2
xxP[3]=xxp3
xxP[4]=xxp4
xxP[5]=xxp5

xx = sym.zeros(1,ns)
xx[0]=x0
xx[1]=x1
xx[2]=x2
xx[3]=x3
xx[4]=x4
xx[5]=x5

uu = sym.zeros(1,ni)
uu[0]=u0
uu[1]=u1

AA = sym.zeros(ns,ns)
BB = sym.zeros(ns,ni)

################- jacobian - #########################
for i in range(ns):
  for j in range(ns):
      AA[i,j] = sym.diff(xxP[i],xx[j])

################ - STATE-INPUT- ##########################

for i in range(ns):
  for j in range(ni):
    BB[i,j] = sym.diff(xxP[i],uu[j])

#################################################

# print(xxP)
# pprint('---------------------------------------')
pprint(AA[3,:])
# pprint('---------------------------------------')
# print(BB)