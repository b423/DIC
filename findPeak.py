import numpy as np
from scipy.optimize import leastsq, fmin
from scipy.interpolate import Rbf
import cv2
import logging
import functools
import fitting
"""
亚像素的装饰器
"""
def sub_loc(func):
    @functools.wraps(func)
    def subpixel(res):
       min_v, max_v, min_loc, max_loc = func(res) 
       max_x, max_y = max_loc
       if res.shape[0] == 1:
           try:
               p, loc = fitting.quadratic_one_fitting(np.arange(max_x-2,max_x+3),res[0,max_x-2:max_x+3])
               best_x, best_y = loc[0], max_y
           except ValueError:
               best_x, best_y = max_loc
       elif res.shape[0] > 20:
           xx = []
           yy = []
           tt = []
           try:
               for  y_loc in range(max_y-2,max_y+3):
                   for  x_loc in range(max_x-2,max_x+3):
                       xx.append(x_loc)
                       yy.append(y_loc)
                       tt.append(res[y_loc, x_loc])
               p, loc = fitting.quadratic_two_fitting(np.array(xx),np.array(yy),np.array(tt))
               best_x, best_y = loc
           except IndexError:
               best_x, best_y = max_loc
       else:
           best_x, best_y = max_loc
       return min_v, max_v, min_loc, (best_x,best_y)
    return subpixel      
           
       



@sub_loc
def minMaxLoc(res, mask = None):
    min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res, mask)
    return min_v, max_v, min_loc, max_loc
    
"""
位移相关测量
"""
def findPeak_LR(img_L, img_R,  xs, ys, shape):
    best_loc_x = np.zeros(xs.shape[0])
    best_loc_y = np.zeros(ys.shape[0])
    best_v = np.zeros(xs.shape[0])
    
    for ii, (x, y) in enumerate(zip(xs.astype(int), ys.astype(int))):
        roi = img_L[y-20 :y + 21, x-20:x+21];

        #target = img_R;
        #res = cv2.matchTemplate(target, roi, 3);
        #min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res);
        #max_loc,v = sub_template_match(res);

        target = img_R[y - 30:y + 31, x - 22:];
        res = cv2.matchTemplate(target, roi, 3);
        min_v, v, min_loc, max_loc = minMaxLoc(res)
        logging.info(v)
        max_loc = (max_loc[0] + x - 22, max_loc[1] + y - 30);
        best_loc_x[ii] = max_loc[0] + 20;
        best_loc_y[ii] = max_loc[1] + 20;
        best_v[ii] = v; 

    return best_loc_x.reshape(shape), best_loc_y.reshape(shape), best_v.reshape(shape);           




"""
时序相关测量
"""

def findPeak_BA(img_B, img_A, xs, ys, shape):
    
    best_loc_x = np.zeros(xs.shape)
    best_loc_y = np.zeros(xs.shape)
    best_v = np.zeros(xs.shape)
    #org_loc_x = np.zeros([len(y_range), len(x_range)]);
    #org_loc_y = np.zeros([len(y_range), len(x_range)]);

    for ii, (x, y) in enumerate(zip(xs.astype(int), ys.astype(int))):
            roi = img_B[y-20 :y + 21, x-20:x+21];
            target = img_A;

            res = cv2.matchTemplate(target, roi, 3);
            min_v, v, min_loc, max_loc = minMaxLoc(res);

            #target = img_R[y - 30:y + 31, x - 22:];
            #res = cv2.matchTemplate(target, roi, 3);
            #max_loc, v = sub_template_match(res);
            #max_loc = (max_loc[0] + x - 22, max_loc[1] + y - 30);

            best_loc_x[ii] = max_loc[0] + 20;
            best_loc_y[ii] = max_loc[1] + 20;
            best_v[ii] = v; 
    return best_loc_x.reshape(shape), best_loc_y.reshape(shape), best_v.reshape(shape);           

