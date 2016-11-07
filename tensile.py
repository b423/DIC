import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#import copy
import cv2
import logging
import findPeak
from disparity import disparity
from reconstruction import reconstruction
import profileFetching
import pickle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
import coordTransform
import strainComputing


logging.basicConfig(
    level = logging.INFO,
)
img_R = cv2.imread(u"R000001.bmp",0);
img_L = cv2.imread(u"L000001.bmp",0);

#plt.subplot(121);
#cv2.rectangle(img_L, (500,500), (700,700), 255, 2);
#plt.imshow(img_L,cmap='gray');
#plt.subplot(122);
#cv2.rectangle(img_R, (500,500), (700,700), 255, 2);
#plt.imshow(img_R,cmap='gray');
#plt.show();

"""
相机参数
"""
#param : cx, cy, cx_, f, tx
param = [-531.977, 135.4114, 3767.9713, 14635, -185.2824]

x_range = np.arange(530, 750, 20)
y_range = np.arange(400, 700, 20)

w = x_range.shape[0]
h = y_range.shape[0]

org_loc_x = np.zeros([len(y_range), len(x_range)]);
org_loc_y = np.zeros([len(y_range), len(x_range)]);
for ii, x in enumerate(x_range):
        for jj, y in enumerate(y_range):                
            org_loc_x[jj, ii] = x;
            org_loc_y[jj, ii] = y;

"""
第一对图像的位置相关的匹配
"""
peaks_x, peaks_y, match_v = findPeak.findPeak_LR(img_L, img_R,\
                                                 org_loc_x.flatten(),
                                                 org_loc_y.flatten(), \
                                                 (h, w))
org_loc_x_R, org_loc_y_R = np.around(peaks_x), np.around(peaks_y)


"""
    3D reconstruction
"""

disp = disparity(org_loc_x.flatten(), org_loc_x_R.flatten())
pos_3D = reconstruction(org_loc_x.flatten(), \
                        org_loc_y.flatten(), \
                        disp,param)

"""
    bad points deletion
"""
z_mean = np.mean(pos_3D[:,2])
z_std = np.std(pos_3D[:,2])
good_pos = []
bad_pos = []
for ii in range(pos_3D.shape[0]): 
    if pos_3D[ii, 2] < z_mean + 2 * z_std and\
        pos_3D[ii, 2] > z_mean - 2 * z_std:
        good_pos.append(pos_3D[ii, :]);
    else:
        bad_pos.append(pos_3D[ii, :]);
good_pos = np.array(good_pos)
bad_pos = np.array(bad_pos)

"""
时序测量
"""
img_L_A = cv2.imread(u"L000104.bmp",0);
img_R_A = cv2.imread(u"R000104.bmp",0);
peaks_x_L_A , peaks_y_L_A , match_v = findPeak.findPeak_BA(img_L, img_L_A,\
                                                            org_loc_x.flatten(),\
                                                            org_loc_y.flatten(),\
                                                            (h,w));

peaks_x_R_A , peaks_y_R_A , match_v = findPeak.findPeak_BA(img_R, img_R_A,\
                                                            org_loc_x_R.flatten(),\
                                                            org_loc_y_R.flatten(),\
                                                            (h,w));

"""
    3D reconstruction
"""

disp_A= disparity(peaks_x_L_A.flatten(), peaks_x_R_A.flatten())
pos_3D_A = reconstruction(peaks_x_L_A.flatten(), \
                        peaks_y_L_A.flatten(), \
                        disp_A,param)

"""
    bad points deletion and 补坏点
"""
z_mean = np.mean(pos_3D_A[:,2]);
z_std = np.std(pos_3D_A[:,2]);
good_pos_A = [];
bad_pos_A = []
for ii in range(pos_3D_A.shape[0]): 
    if pos_3D_A[ii, 2] < z_mean + 2 * z_std and\
        pos_3D_A[ii, 2] > z_mean - 2 * z_std:
        good_pos_A.append(pos_3D_A[ii, :]);
    else:
        bad_pos_A.append(pos_3D_A[ii, :]);
        
good_pos_A = np.array(good_pos_A);
bad_pos_A = np.array(bad_pos_A);
logging.info(bad_pos_A.shape);

"""
位移和应变
"""
#strainComputing.strain_computing_global(pos_3D_A, pos_3D, (h,w))




"""
Display
"""

#xyz = np.array([[ii, n * ii + y0, p * ii + z0] for ii in np.arange(- 250,-180)]);

show_L = cv2.cvtColor(img_L, cv2.COLOR_GRAY2RGB);
show_R = cv2.cvtColor(img_R, cv2.COLOR_GRAY2RGB);

for x,y in zip(org_loc_x.flatten().astype(int), org_loc_y.flatten().astype(int)):
    cv2.circle(show_L, (x, y), 10,(255,0,0),2);
        
for x, y in zip(org_loc_x_R.flatten().astype(int), org_loc_y_R.flatten().astype(int)):
    cv2.circle(show_R, (x, y), 10,(255,0,0),2);
   
"""
fig = plt.figure(figsize=plt.figaspect(2.))
fig.suptitle('A tale of 2 subplots')
ax = fig.add_subplot(2, 2, 1);

org = ax.imshow(show_L);#plot(t1, f(t1), 'bo',t2, f(t2), 'k--', markerfacecolor='green')
ax = fig.add_subplot(2, 2, 2);
peak = ax.imshow(show_R);

ax = fig.add_subplot(2, 2, 3, projection = '3d');
ax.ticklabel_format(useOffset=False)
ax.set_xlabel('X ');
ax.set_ylabel('Y ');
ax.set_zlabel('Z ');
sur = ax.scatter(good_pos[:,0], good_pos[:,1], good_pos[:,2]);
""" 

#plt.show();

show_L_A = cv2.cvtColor(img_L_A, cv2.COLOR_GRAY2RGB);
for x, y in zip(peaks_x_L_A.flatten().astype(int),\
                 peaks_y_L_A.flatten().astype(int)):
    cv2.circle(show_L_A, (x,y), 10,(255,0,0),2);

show_R_A = cv2.cvtColor(img_R_A, cv2.COLOR_GRAY2RGB);
for x, y in zip(peaks_x_R_A.flatten().astype(int), \
                peaks_y_R_A.flatten().astype(int)):
    cv2.circle(show_R_A, (x,y), 10,(255,0,0),2);

fig2 = plt.figure(figsize=plt.figaspect(2.))
fig2.suptitle('A tale of 2 subplots')

ax = fig2.add_subplot(2,3,1);
org = ax.imshow(show_L);#plot(t1, f(t1), 'bo',t2, f(t2), 'k--', markerfacecolor='green')

ax = fig2.add_subplot(2,3,2);
peak = ax.imshow(show_R);show_R

ax = fig2.add_subplot(2,3,3, projection = '3d');
ax.ticklabel_format(useOffset=False)
ax.set_xlabel('X ');
ax.set_ylabel('Y ');
ax.set_zlabel('Z ');
sur = ax.scatter(good_pos[:,0], good_pos[:,1], good_pos[:,2]);

ax = fig2.add_subplot(2,3,4);
org = ax.imshow(show_L_A);#plot(t1, f(t1), 'bo',t2, f(t2), 'k--', markerfacecolor='green')

ax = fig2.add_subplot(2,3,5);
peak = ax.imshow(show_R_A);

ax = fig2.add_subplot(2,3,6, projection = '3d');
ax.ticklabel_format(useOffset=False)
ax.set_xlabel('X ');
ax.set_ylabel('Y ');
ax.set_zlabel('Z ');
sur = ax.scatter(good_pos_A[:,0], good_pos_A[:,1], good_pos_A[:,2]);

plt.show()
