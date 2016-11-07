import numpy as np
from scipy import optimize
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
import fitting 
"""
inputs: 
        xyz:current locations of mesh
        xyz0: origin locations of mesh
"""
def strain_computing_single(xyz,xyz0):
    """
    build three cubic functions:
    u = fx(x0, y0, z0)
    v = fy(x0, y0, z0)
    w = fz(x0, y0, z0)
    """
    ###局部三元三次曲面拟合位移
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    x0, y0, z0 = xyz0[:,0],xyz0[:,1], xyz0[:,2]
    u, v, w = x-x0, y-y0, z-z0

    p_w,_ = fitting.tri_three_fitting(x0,y0,z0,w)
    p_v,_ = fitting.tri_three_fitting(x0,y0,z0,v)
    p_u,_ = fitting.tri_three_fitting(x0,y0,z0,u)
    
    x_a, y_a, z_a = x0[12], y0[12], z0[12]
    for x_a, y_a, z_a in zip(x0, y0, z0):
        ###柯西——格林应变公式
        exx = 3 * x_a ** 2 * p_u[0] + 2 * x_a * y_a * p_u[3] + y_a ** 2 * p_u[4] + \
            2 * x_a * z_a * p_u[5]  + z_a**2*p_u[6] + y_a*z_a*p_u[9] +\
            2*x_a*p_u[10] + y_a*p_u[13] + z_a*p_u[14] + p_u[16]

        eyx = 3 * x_a ** 2 * p_v[0] + 2 * x_a * y_a * p_v[3] + y_a ** 2 * p_v[4] + \
            2 * x_a * z_a * p_v[5]  + z_a**2*p_v[6] + y_a*z_a*p_v[9] +\
            2*x_a*p_v[10] + y_a*p_v[13] + z_a*p_v[14] + p_v[16]

        ezx = 3 * x_a ** 2 * p_w[0] + 2 * x_a * y_a * p_w[3] + y_a ** 2 * p_w[4] + \
            2 * x_a * z_a * p_w[5]  + z_a**2*p_w[6] + y_a*z_a*p_w[9] +\
            2*x_a*p_w[10] + y_a*p_w[13] + z_a*p_w[14] + p_w[16]
        ###
        eyy = 3 * y_a ** 2 * p_v[1] + x_a**2*p_v[3] + 2*x_a*y_a*p_v[4] +\
            2*y_a*z_a*p_v[7] + z_a**2*p_v[8] + x_a*z_a*p_v[9]+\
            2*y_a*p_v[11] + x_a*p_v[13] + z_a*p_v[15] + p_v[17]
        
        exy = 3 * y_a ** 2 * p_u[1] + x_a**2*p_u[3] + 2*x_a*y_a*p_u[4] +\
            2*y_a*z_a*p_u[7] + z_a**2*p_u[8] + x_a*z_a*p_u[9]+\
            2*y_a*p_u[11] + x_a*p_u[13] + z_a*p_u[15] + p_u[17]

        ezy = 3 * y_a ** 2 * p_w[1] + x_a**2*p_w[3] + 2*x_a*y_a*p_w[4] +\
            2*y_a*z_a*p_w[7] + z_a**2*p_w[8] + x_a*z_a*p_w[9]+\
            2*y_a*p_w[11] + x_a*p_w[13] + z_a*p_w[15] + p_w[17]
        ###
        ezz = 3*z_a**2*p_w[2] + x_a**2*p_w[5] + 2*x_a*z_a*p_w[6] + \
            y_a**2*p_w[7] + 2*y_a*z_a*p_w[8] + x_a*y_a*p_w[9] +\
            2*z_a*p_w[12] + x_a*p_w[14] + y_a * p_w[15] + p_w[18]

        exz = 3*z_a**2*p_u[2] + x_a**2*p_u[5] + 2*x_a*z_a*p_u[6] + \
            y_a**2*p_u[7] + 2*y_a*z_a*p_u[8] + x_a*y_a*p_u[9] +\
            2*z_a*p_u[12] + x_a*p_u[14] + y_a * p_u[15] + p_u[18]
        
        eyz = 3*z_a**2*p_v[2] + x_a**2*p_v[5] + 2*x_a*z_a*p_v[6] + \
            y_a**2*p_v[7] + 2*y_a*z_a*p_v[8] + x_a*y_a*p_v[9] +\
            2*z_a*p_v[12] + x_a*p_v[14] + y_a * p_v[15] + p_v[18]

        exy = 0.5 * (exy+eyz)
        exz = 0.5 * (exz+ezx)
        eyz = 0.5 * (eyz+ezy)
        return np.array([exx, eyy, ezz, exy, exz, eyz])

def main_strain(strain):
    #应变不变量
    I1 = np.sum(strain[0:3])
    I2 = strain[0]*strain[1] + strain[1]*strain[2] + strain[2]*strain[0] -\
        strain[3]**2 - strain[4]**2 -strain[5]**2
    I3 = strain[0]*strain[1]*strain[2] + 2*strain[3]*strain[4]*strain[5] -\
        strain[0]*strain[5]**2 - strain[1]*strain[4]**2 - strain[2]*strain[3]**2
    
    m_strains = np.zeros([3,1])
    b,c,d = -I1,I2,-I3
    A = b**2 - 3*c
    B = b*c - 9*d
    C = c**2 - 3*b*d
    delta = B**2 - 4*A*C
    #盛金公式求一元三次方程
    try:
        if A == B and B == 0:
            m_strains = np.array([-b/3,-c/b,-3*d/c])
        elif delta == 0.:
            m_strains = np.array([-b + B/A,-0.5 * B/A,-0.5*B/A])
        elif delta < 0.:
            T = (2*A*b-3*B)/(2*A**(3/2))
            theta = np.arccos(T)
            x1 = (-b-2*A**0.5*np.cos(theta/3))/3
            x2 = (-b+A**0.5*(np.cos(theta/3)+ 3**0.5*np.sin(theta/3)))/3
            x3 = (-b+A**0.5*(np.cos(theta/3)- 3**0.5*np.sin(theta/3)))/3
            m_strains = np.array([x1, x2, x3])
        else:
            raise ValueError
    except ValueError as e:
        logging.info(e)
    logging.info(m_strains)



def strain_computing_global(xyz, xyz0, shape):
    """
    利用前后两个状态的三维坐标求应变
    """
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    x0, y0, z0 = xyz0[:,0],xyz0[:,1], xyz0[:,2]
    x0_mesh = x0.reshape(shape)
    y0_mesh = y0.reshape(shape)
    z0_mesh = z0.reshape(shape)

    x_mesh = x.reshape(shape)
    y_mesh = y.reshape(shape)
    z_mesh = z.reshape(shape)
    
    main_strains = np.zeros([shape[0],shape[1],3])
    for jj in range(2,shape[0]-2):
        for ii in range(2,shape[1] - 2):
            x_l = x_mesh[jj - 2: jj + 3, ii-2:ii+3].reshape(-1,1)
            y_l = y_mesh[jj - 2: jj + 3, ii-2:ii+3].reshape(-1,1)
            z_l = z_mesh[jj - 2: jj + 3, ii-2:ii+3].reshape(-1,1)

            x0_l = x0_mesh[jj - 2: jj + 3, ii-2:ii+3].reshape(-1,1)
            y0_l = y0_mesh[jj - 2: jj + 3, ii-2:ii+3].reshape(-1,1)
            z0_l = z0_mesh[jj - 2: jj + 3, ii-2:ii+3].reshape(-1,1)
            strain = strain_computing_single(np.hstack([x_l, y_l, z_l]),np.hstack([x0_l, y0_l, z0_l]))
            main_strain(strain) 

    
    logging.info(strains)
