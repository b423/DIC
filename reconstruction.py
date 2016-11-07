import numpy as np
import logging

def reconstruction(loc_left_x, loc_left_y, disp, param): 
   
    cx, cy, cx_, f, tx = param;
    Q = np.eye(4);
    Q[0, 3], Q[1, 3], Q[2, 3], Q[3,2], Q[3,3], Q[2, 2] =  -cx, -cy, f, -1 / tx, (cx - cx_) / tx, 0;
    
    loc_disp = np.array([(x, y, d, 1) for (x, y, d) \
                                    in zip(loc_left_x, loc_left_y, disp)]);
    pos = np.dot(Q, loc_disp.T);
    pos = (pos / pos[3,:])[0:3,:].T;
    #logging.info(pos);
    return pos;
                
    