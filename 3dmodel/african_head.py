import numpy as np
import re
from PIL import Image as img
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import african_head_func as ahf

N = 1024
canvas1 = np.zeros((N, N, 3), dtype=np.uint8)
canvas1[:, :] = [0, 0, 0]
canvas2 = np.zeros((N, N, 3), dtype=np.uint8)
canvas2[:, :] = [0, 0, 0]
canvas3 = np.zeros((N, N, 3), dtype=np.uint8)
canvas3[:, :] = [0, 0, 0]
canvas4 = np.zeros((N, N, 3), dtype=np.uint8)
canvas4[:, :] = [0, 0, 0]
images = []
c = [-1, 0, -1]
koef_s = 0.9
ang_x, ang_y, ang_z = 8*np.pi/180, 12*np.pi/180, 16*np.pi/180

z_buf = np.ones((N, N), dtype=float)
p_cam_A = [2, 2, 2]
p_cam_B = [-2, -2, 0]

texture_img = img.open('african_head_diffuse.tga')
texture_img = texture_img.rotate(180, expand=True)
faces = set()
with open('african_head.obj', 'r') as f:
    for line in f:
        if 'f ' in line:
            a = line.replace('/', ' ').split()
            del a[0]
            faces.add(((int(a[0]),int(a[1]),int(a[2])),(int(a[3]),int(a[4]),int(a[5])),(int(a[6]),int(a[7]),int(a[8]))))

faces_v_vt_vn = list()
with open('african_head.obj', 'r') as f:
    L = f.readlines()
    for face in faces:
        v1_ind, v2_ind, v3_ind = face[0][0]-1, face[1][0]-1, face[2][0]-1
        v1, v2, v3 = L[v1_ind].split(), L[v2_ind].split(), L[v3_ind].split()
        del v1[0]
        del v2[0]
        del v3[0]
        v1, v2, v3 = [float(i) for i in v1], [float(i) for i in v2], [float(i) for i in v3]

        vt1_ind, vt2_ind, vt3_ind = face[0][1] + 1260 - 1, face[1][1] + 1260 - 1, face[2][1] + 1260 - 1
        vt1, vt2, vt3 = L[vt1_ind].split(), L[vt2_ind].split(), L[vt3_ind].split()
        del vt1[0]
        del vt2[0]
        del vt3[0]
        vt1, vt2, vt3 = [float(i) for i in vt1], [float(i) for i in vt2], [float(i) for i in vt3]

        vn1_ind, vn2_ind, vn3_ind = face[0][2]+1260+1339+2-1, face[1][2]+1260+1339+2-1, face[2][2]+1260+1339+2-1
        vn1, vn2, vn3 = L[vn1_ind].split(), L[vn2_ind].split(), L[vn3_ind].split()
        del vn1[0]
        del vn2[0]
        del vn3[0]
        vn1, vn2, vn3 = [float(i) for i in vn1], [float(i) for i in vn2], [float(i) for i in vn3]
        faces_v_vt_vn.append(((v1, v2, v3), (vt1, vt2, vt3), (vn1, vn2, vn3)))

d = 24
a = 360 / d
k = 360 / d
ang = np.pi / 180
p_cam_A = [3, 4, 3]
p_cam_B = [0, 0, 0]
c = [0, 0, 3]
for i in range(d):
    z_buf = np.ones((N, N), dtype=float)
    z_buf_im = ahf.z_buf_im(z_buf, faces_v_vt_vn, c, koef_s, a*ang, ang, 90*ang, p_cam_A, p_cam_B, N, 2)
    canvas4[:, :] = [0, 0, 0]
    for face in faces_v_vt_vn:
        X1 = np.matrix([face[0][0][0], face[0][0][1], face[0][0][2], 1])
        X2 = np.matrix([face[0][1][0], face[0][1][1], face[0][1][2], 1])
        X3 = np.matrix([face[0][2][0], face[0][2][1], face[0][2][2], 1])

        X1_t = np.matrix([[face[1][0][0]], [face[1][0][1]], [face[1][0][2]]])
        X2_t = np.matrix([[face[1][1][0]], [face[1][1][1]], [face[1][1][2]]])
        X3_t = np.matrix([[face[1][2][0]], [face[1][2][1]], [face[1][2][2]]])

        X1_xy = ahf.new_X(X1, c, koef_s, a*ang, ang, 90*ang, p_cam_A, p_cam_B, N, 2)
        X2_xy = ahf.new_X(X2, c, koef_s, a*ang, ang, 90*ang, p_cam_A, p_cam_B, N, 2)
        X3_xy = ahf.new_X(X3, c, koef_s, a*ang, ang, 90*ang, p_cam_A, p_cam_B, N, 2)

        if ahf.back_face_culling(p_cam_A, X1_xy, X2_xy, X3_xy) < 0:
            ahf.rectangle(3, z_buf_im, canvas4, texture_img, X1_xy, X2_xy, X3_xy, X1_t, X2_t, X3_t)
    a += k
    im_ani = img.fromarray(canvas4)
    images.append(im_ani)

images[0].save('african_head.gif', save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)






