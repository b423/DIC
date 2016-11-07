import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#import copy
import cv2
import logging
from findPeak import findPeak
from disparity import disparity
from reconstruction import reconstruction
import profileFetching
import pickle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
import coordTransform


logging.basicConfig(
    level = logging.INFO,
)
img_R = cv2.imread(u"L000001.bmp",0);
img_L = cv2.imread(u"R000001.bmp", 0);

#plt.subplot(121);
#cv2.rectangle(img_L, (500,500), (700,700), 255, 2);
#plt.imshow(img_L,cmap='gray');
#plt.subplot(122);
#cv2.rectangle(img_R, (500,500), (700,700), 255, 2);
#plt.imshow(img_R,cmap='gray');
#plt.show();


x_range = [ii for ii in range(550, 1100, 10)];
y_range = [ii for ii in range(290, 750, 20)];

org_loc_x = np.zeros([len(y_range), len(x_range)]);
org_loc_y = np.zeros([len(y_range), len(x_range)]);
for ii, x in enumerate(x_range):
        for jj, y in enumerate(y_range):                
            org_loc_x[jj, ii] = x;
            org_loc_y[jj, ii] = y;
org_loc_x = org_loc_x.flatten();
org_loc_y = org_loc_y.flatten();
"""
best matched position with sub-pixel
"""
cal_range = (x_range, y_range);
peaks_x, peaks_y, match_v = findPeak(img_L, img_R, cal_range);
peaks_x = peaks_x.flatten()


"""
3D reconstruction
"""
#param : cx, cy, cx_, f, tx
param = [2420.4, 553.101, 430.3106, 13166, 175.9561];
disp = disparity(org_loc_x, peaks_x);
pos_3D = reconstruction(org_loc_x, org_loc_y, disp,param);

"""
    bad points deletion
"""
z_mean = np.mean(pos_3D[:,2]);
z_std = np.std(pos_3D[:,2]);
logging.info([z_mean, z_std]);
good_pos = [];
bad_pos = []
for ii in range(pos_3D.shape[0]): 
    if pos_3D[ii, 2] < z_mean + 2 * z_std and\
        pos_3D[ii, 2] > z_mean - 2 * z_std:
        good_pos.append(pos_3D[ii, :]);
    else:
        bad_pos.append(pos_3D[ii, :]);
        
good_pos = np.array(good_pos);
bad_pos = np.array(bad_pos);
#logging.info(bad_pos.shape);


"""
axis profile
"""                  
 #待优化参数：
    #   轴线的方向向量 ： m = 1， n， p 
    #   轴线的店：x0 = 0 ， y0， z0   
"""
center_line = profileFetching.fectch_centerline(good_pos, z_mean);
with open("guan.dat", "wb") as fp:
    pickle.dump(center_line, fp);
"""
center_line = [];
with open("guan.dat", "rb") as fb:
    unpickler = pickle.Unpickler(fb);
    center_line = unpickler.load();


m, n, p = 1, center_line[0], center_line[1];
x0, y0, z0 = 0, center_line[2], center_line[3];
xyz = np.array([[ii, n * ii + y0, p * ii + z0] for ii in np.arange(-160,-110)]);

xyz_tr = coordTransform.move_Y(-y0, xyz);
xyz_tr = coordTransform.move_Z(-z0, xyz_tr);
xyz_tr = coordTransform.rotate_Y(-np.arctan(n),xyz_tr);
xyz_tr = coordTransform.rotate_Z(-np.arctan(p),xyz_tr);

good_trans = coordTransform.move_Y(-y0, good_pos);
good_trans = coordTransform.move_Z(-z0, good_trans);
good_trans = coordTransform.rotate_Y(-np.arctan(n), good_trans);
good_trans = coordTransform.rotate_Z(-np.arctan(p),good_trans);

logging.info([y0, z0]);
displace = np.array([[p[0],np.sqrt(p[1] ** 2 + p[2] ** 2)] for p in good_trans]);
h_mean = np.mean(displace[:,1]);
h_std = np.std(displace[:, 1]);
good_profile = [];
for ii in range(displace.shape[0]): 
    if displace[ii, 1] < h_mean + h_std and\
        displace[ii, 1] > h_mean - h_std:
        good_profile.append(displace[ii, :]); 
        
good_profile = np.array(good_profile);

curve = profileFetching.profile_curve(good_profile);
a, b, c, h = curve;
cosine_curve = np.array([[xx,a * np.cos(b * (xx - c)) + h]for xx in np.arange(-200,-100)]); 
logging.info(cosine_curve.shape)
"""
Display
"""

#xyz = np.array([[ii, n * ii + y0, p * ii + z0] for ii in np.arange(- 250,-180)]);

show_L = cv2.cvtColor(img_L, cv2.COLOR_GRAY2RGB);
show_R = cv2.cvtColor(img_R, cv2.COLOR_GRAY2RGB);

for x,y in zip(org_loc_x.flatten(), org_loc_y.flatten()):
        cv2.circle(show_L, (int(x), int(y)), 10,(255,0,0),2);
        
for x, y in zip(peaks_x.flatten(), peaks_y.flatten()):
    cv2.circle(show_R, (int(x), int(y)), 10,(255,0,0),2);
   
 
fig = plt.figure(figsize=plt.figaspect(2.))
fig.suptitle('A tale of 2 subplots')
ax = fig.add_subplot(2, 2, 1);

org = ax.imshow(show_L);#plot(t1, f(t1), 'bo',t2, f(t2), 'k--', markerfacecolor='green')
ax = fig.add_subplot(2, 2, 2);
peak = ax.imshow(show_R);
ax = fig.add_subplot(2, 2, 3, projection = '3d');
ax.ticklabel_format(useOffset=False)
sur = ax.scatter(good_pos[:,0], good_pos[:,1], good_pos[:,2]);
ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], 'r-');
ax.set_xlim(-155,-115);
ax.set_ylim([-25, 25]);
ax.set_zlim([-50, 0]);
ax.set_xlabel('X ');
ax.set_ylabel('Y ');
ax.set_zlabel('Z ');

ax = fig.add_subplot(2, 2, 4);
ax.scatter(good_profile[:,0], good_profile[:,1]);
ax.plot(cosine_curve[:,0], cosine_curve[:,1], 'r-');
ax.set_xlim(-155,-115);
ax.set_ylim(0,40);

plt.show();
