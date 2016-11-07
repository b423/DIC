import numpy as np
"""
angle in degree and 3d points 
"""
def rotate_Z(angle, points):
    mat = np.eye(3);
    mat[0,0] = np.cos(angle);
    mat[0,2] = np.sin(angle);
    mat[2,2] = np.cos(angle);
    mat[2,0] = -np.sin(angle);
    points_trans = np.dot(points, mat);
    return points_trans;
     
def rotate_Y(angle, points):
    mat = np.eye(3);
    mat[0,0] = np.cos(angle);
    mat[0,1] = np.sin(angle);
    mat[1,1] = np.cos(angle);
    mat[1,0] = -np.sin(angle);
    points_trans = np.dot(points, mat);
    return points_trans; 
    
def move_Z (displace, points):
    points[:,2] = points[:,2] + displace;
    return points
    
def move_Y (displace, points):
    points[:,1] = points[:,1] + displace;
    return points;

