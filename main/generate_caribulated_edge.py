import numpy as np
import trimesh
import pyrender
import cv2
import edge_generate as eg
import camera_calibration as cc
import camera_target as ct

def calibrate(qr_size,camera_height):


    target_area = ct.main()

    optimal_yfov, optimal_area = cc.find_optimal_yfov(target_area, qr_size, camera_height)
    print(f"Optimal yfov: {optimal_yfov}, Area: {optimal_area}")
    return optimal_yfov,optimal_area
if __name__ == '__main__':

    qr_size = (42,42)
    camera_height = 320
    obj_name = "TH-33"
    pattern = "SIDE"
    r = 0

    #はじめてはここのコメントアウトを解除する
    '''y_fov, area = calibrate(qr_size,camera_height)
    scale = np.sqrt((qr_size[0]*qr_size[1])/area)
    print(f"Scale;{scale}")'''
    
    y_fov = 1.0413765843976148
    scale = 0.21761658031088082


    obj_list = ["PS-10SH","PS-18SU","PS-24SU","PS-33SU","TH-10","TH-18","TH-24","TH-33"]
    pattern_list = ["FRONT","BACK","SIDE"]
    x_values = [200, 100, 50, 0, -50, -100, -200]
    y_values = [150, 50, 20, 0, -20, -50, -150]
    r_values = np.linspace(-180,180,37)[:-1]

    eg.render_grid2(obj_list, pattern_list, r_values, scale, camera_height, light_intensity=200, yfov=y_fov, num=1000000)