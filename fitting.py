import numpy as np
from scipy import optimize
import logging


def _func(p, x):
    f = np.dot(x,p)
    return f

def _diff(p,y,x):
    return y - _func(p,x)

def quadratic_one_fitting(x, t):
    p0 = np.zeros([3,1])
    a0 = x ** 2
    a1 = x
    a2 = np.ones(x.shape)
    args_x = np.vstack([a0,a1,a2]).T
    p = optimize.leastsq(_diff,p0,args=(t,args_x))[0]
    pole_loc = np.array([- p[1] / (2 * p[0]),])
    return p, pole_loc 

def quadratic_two_fitting(x,y,t):
    p0 = np.zeros([6,1], dtype=float)
    a0 = x ** 2
    a1 = y ** 2
    a2 = x * y
    a3 = x
    a4 = y
    a5 = np.ones(x.shape)

    args_x = np.vstack([a0,a1,a2,a3,a4,a5]).T
    p = optimize.leastsq(_diff,p0,args=(t,args_x))[0]
    x_b = (2*p[1]*p[3]-p[4]*p[2])/(p[2]**2-4*p[0]*p[1])
    y_b = (2*p[0]*p[4]-p[2]*p[3])/(p[2]**2-4*p[0]*p[1])
    pole_loc = np.array([x_b,y_b])
    return p, pole_loc

def tri_three_fitting(x0,y0,z0,t):
    p0 = np.zeros([20,1])
    """
    three degree 
    """
    a0 = x0 ** 3
    a1 = y0 ** 3
    a2 = z0 ** 3
    a3 = x0 ** 2 * y0
    a4 = x0 * y0 ** 2
    a5 = x0 ** 2 * z0
    a6 = x0 * z0 ** 2
    a7 = y0 ** 2 * z0
    a8 = y0 * z0 ** 2
    a9 = x0 * y0 * z0

    """
    two degree 
    """
    a10 = x0 ** 2
    a11 = y0 ** 2
    a12 = z0 ** 2
    a13 = x0 * y0
    a14 = x0 * z0
    a15 = y0 * z0

    """
    one degree 
    """
    a16 = x0
    a17 = y0
    a18 = z0

    """
    constant degree 
    """
    a19 = np.ones(x0.shape)
    
    args_x = np.vstack([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,\
                    a10,a11,a12,a13,a14,a15,a16,a17,\
                    a18,a19]).T
    p = optimize.leastsq(_diff,p0,args=(t.T,args_x))[0]
    return p, (0,0)

