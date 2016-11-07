import numpy as np
from scipy.optimize import differential_evolution, leastsq, least_squares, minimize
import logging

def fectch_centerline(pos_3d = None,z_mean = 2000):    
   """
   求管坯轴线
   """ 
    
    def func(p, x):
        """
        a, b, c, d, e, f, g, h, i, j = p;
        u, v  = x;
        u, v = np.array(u), np.array(v);
        return a*u*u*u + b*v*v*v + c*u*u*v + d*u*v*v + 
                   e*u*u + f*v*v + g*u*v + h*u + i*v + j;    
        """  
        
        n, p, y0, z0 = p;
        m = 1; 
        xx, yy, zz = np.array(x);
        v = np.array([m, n, p]);
        v_x, v_y, v_z = xx, yy - y0, zz - z0;
        v_m = np.array([v_x, v_y, v_z]).T;
        
        #t = np.cross(v, v_m);
        #v_up = np.array([np.sqrt(np.dot(tt, tt)) for tt in t]);
        #down = np.sqrt(np.dot(v,v));
        #disp = v_up /down;
        
        v_up = (p * (yy - y0) - n * (zz - z0)) ** 2 + \
                (m * (zz - z0) - p * xx) ** 2 + \
                (n * xx - m * (yy - y0)) ** 2;
        down = np.dot(v,v);
        disp = np.sqrt(v_up /down);
        res = np.var(disp);
        #logging.info(disp);
        return res;
        
       
         
    #def errors(p, x, y):
    #    return func(p,x) - y;         
    
    #try:
    #    res = leastsq(errors, p0 , args = ([pos_3d,], y))[0];
    #        #y_value should be equal to the max_loc for reconstruction
    #except TypeError as e:
    #        logging.debug(e)
    #p0 = [1,1,1,1,1, -1400, 5, 1.];
    #y = np.ones(pos_3d.shape[0]); 
    #Ares = leastsq(func, [1, 0, 0 , 0, 0],\
    #              args =([pos_3d[:,0], pos_3d[:,1], pos_3d[:,2]],y));
    #cons_line = (#{'type': 'eq','fun' : lambda x: np.array([x[0]**2 + x[1]**2 + x[2] ** 2 - 1])},\
                #{'type': 'ineq', 'fun': lambda x: np.abs(x[0])- np.abs(x[1])},\
                #{'type': 'ineq', 'fun': lambda x: np.abs(x[0]) - np.abs(x[2])},\
                #{'type': 'ineq', 'fun': lambda x: x[0]},\
                #{'type': 'ineq', 'fun': lambda x: 2500 - x[4]}
    #            );
    #logging.info(pos_3d[:,1]);
    #待优化参数：
    #   轴线的方向向量 ： m = 1， n， p 
    #   轴线的店：x0 = 0 ， y0， z0   
    res = minimize(func, x0 = [ 0, 0, -20, z_mean + 800],\
                args = ([pos_3d[:,0], pos_3d[:,1], pos_3d[:,2]]), \
    #            constraints = cons_line,\
        	method='Nelder-Mead', options={'disp': True });
    #res = differential_evolution(func, bounds = [(-1, 1),(-1, 1),(-50, 50),(800, 1500)],\
    #           args = ([pos_3d[:,0], pos_3d[:,1], pos_3d[:,2]],), \
    #           #constraints = cons_line,\
    #   	#method='SLSQP', \
    #            disp=True);
    #logging.info(res.x);
    return res.x;
    
def profile_curve(pts, method = 'cosine'): 
    """
    轮廓拟合
    """
    x_min, x_max = np.min(np.abs(pts[:,0])), np.max(np.abs(pts[:,0]));
    h0 = np.mean(pts[:,1])
    
    def cosine (p, x):
        a, b, c, h = p;
        x, y = pts[:,0], pts[:,1];
        return np.sum((y - (a * np.cos(b * (x - c)) + h))**2);
    
    if method == 'cosine':           
        cons_line = ({'type': 'ineq', 'fun': lambda x: x[0]},
                    {'type': 'ineq', 'fun': lambda x: x[1]},
                    {'type': 'ineq', 'fun': lambda x: x[2] - x_min},
                    {'type': 'ineq', 'fun': lambda x: x_max - x[2]});
        res = minimize(cosine, x0 = [ 1, 0.1, (x_min+ x_max) / 2, h0],\
                        args = ([pts[:,0], pts[:,1]]), \
                        constraints = cons_line,\
        	            method='SLSQP', options={'disp': True });
                        
        logging.info(res);
        return res.x;
    
    """
    def func(p, x):
        a, b, c, d, e, f, g, h, i, j = p;
        u, v  = x;
        u, v = np.array(u), np.array(v);
        return a*u*u*u + b*v*v*v + c*u*u*v + d*u*v*v + \
                   e*u*u + f*v*v + g*u*v + h*u + i*v + j;     
    """       
            
   # def error(p, x, y):
   #     return func(p, x) - y;
        
    #p0 = [1,1,1,1,1,1];
    #    try:
    #        a, b, c, d, e, f, g, h, i, j = leastsq(error, p0, args = ([u,v], y))[0];
    #        #y_value should be equal to the max_loc for reconstruction
    #        best_loc = ((2 * b * d - c * e)/(c * c - 4 * a * b), max_loc[1]);
    #        logging.debug(np.array(best_loc) - np.array(max_loc));
    ##    except TypeError as e:
    #        logging.debug(e)
    #        best_loc = max_loc;        