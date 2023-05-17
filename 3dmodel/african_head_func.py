import numpy as np
from PIL import Image as img
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

def draw_pixel(x, y, array, color):
    x = int(x)
    y = int(y)
    array[x, y] = color
    return array

def set_pixel(x, y, canvas, color):
    if x < 0:
        x = -round(x)
    else:
        x = round(x)
    if y < 0:
        y = -round(y)
    else:
        y = round(y)
    draw_pixel(x, y, canvas, color)

def draw_line(x1, y1, x2, y2, canvas, color):
    dx = x2 - x1
    dy = y2 - y1
    sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
    sign_y = 1 if dy > 0 else -1 if dy < 0 else 0
    if dx < 0:
        dx = -dx
    if dy < 0:
        dy = -dy
    if dx > dy:
        pdx, pdy = sign_x, 0
        es, el = dy, dx
    else:
        pdx, pdy = 0, sign_y
        es, el = dx, dy
    x, y = x1, y1
    error, t = el / 2, 0
    set_pixel(x, y, canvas, color)
    while t < el:
        error -= es
        if error < 0:
            error += el
            x += sign_x
            y += sign_y
        else:
            x += pdx
            y += pdy
        t += 1
        set_pixel(x, y, canvas, color)

def find_Mo2w(c, koef_s, ang_x, ang_y, ang_z):
    T = np.matrix([[1, 0, 0, c[0]], [0, 1, 0, c[1]], [0, 0, 1, c[2]], [0, 0, 0, 1]])
    Rx = np.matrix(
        [[1, 0, 0, 0], [0, np.cos(ang_x), -np.sin(ang_x), 0], [0, np.sin(ang_x), np.cos(ang_x), 0], [0, 0, 0, 1]])
    Ry = np.matrix(
        [[np.cos(ang_y), 0, np.sin(ang_y), 0], [0, 1, 0, 0], [-np.sin(ang_y), 0, np.cos(ang_y), 0], [0, 0, 0, 1]])
    Rz = np.matrix(
        [[np.cos(ang_z), -np.sin(ang_z), 0, 0], [np.sin(ang_z), np.cos(ang_z), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    S = np.matrix([[koef_s, 0, 0, 0], [0, koef_s, 0, 0], [0, 0, koef_s, 0], [0, 0, 0, 1]])
    Mo2w = T @ Rx @ Ry @ Rz @ S
    return Mo2w

def find_Mw2c(p_cam_A, p_cam_B):
    Tc = np.matrix([[1, 0, 0, -p_cam_A[0]], [0, 1, 0, -p_cam_A[1]], [0, 0, 1, -p_cam_A[2]], [0, 0, 0, 1]])
    gamma = np.array([p_cam_A[0] - p_cam_B[0], p_cam_A[1] - p_cam_B[1], p_cam_A[2] - p_cam_B[2]])
    g_l = np.sqrt(gamma[0] ** 2 + gamma[1] ** 2 + gamma[2] ** 2)
    gamma = gamma/g_l
    betta = np.array([0, 1, 0]) - gamma[1]*gamma
    alpha = np.cross(betta, gamma)
    # g_l, b_l, a_l длины векторов
    b_l = np.sqrt(betta[0] ** 2 + betta[1] ** 2 + betta[2] ** 2)
    a_l = np.sqrt(alpha[0] ** 2 + alpha[1] ** 2 + alpha[2] ** 2)
    betta = betta/b_l
    alpha = alpha/a_l
    gamma_T, alpha_T, betta_T = gamma[np.newaxis, :].T, alpha[np.newaxis, :].T, betta[np.newaxis, :].T
    Rc = np.matrix([[float(alpha_T[0]), float(betta_T[0]), float(gamma_T[0]), 0],
                    [float(alpha_T[1]), float(betta_T[1]), float(gamma_T[1]), 0],
                    [float(alpha_T[2]), float(betta_T[2]), float(gamma_T[2]), 0], [0, 0, 0, 1]])
    Mw2c = Rc @ Tc
    return Mw2c

def find_Mproj(case_proj):
    if case_proj == 1:
        l, r, b, t, n, f = -4, 0, -2, 2, -10, 10
    if case_proj == 2:
        l, r, b, t, n, f = -8, 8, -8, 8, -10, 10
    '''Mproj=np.matrix([[2*n/(r-l), 0, (r+l)/(r-l), 0],
                     [0, 2*n/(t-b), (t+b)/(t-b), 0],
                     [0, 0, -(f+n)/(f-n), -2*f*n/(f-n)],
                     [0, 0, -1, 0]])'''
    Mproj = np.matrix([[2 / (r - l), 0, 0, -(r+l)/(r-l)],
                       [0, 2 / (t - b), 0, -(t+b)/(t-b)],
                       [0, 0, -2 / (f - n), -(f+n) / (f - n)],
                       [0, 0, 0, 1]])
    return Mproj

def Mviewport(x_l_corn, y_l_corn, N, x_n, y_n, z_n):
    o_x = x_l_corn + N/2
    o_y = y_l_corn + N/2
    x_w = (N/2)*float(x_n) + o_x
    y_w = (N/2) * float(y_n) + o_y
    return x_w, y_w, float(z_n)

def new_X(X, c, koef_s, ang_x, ang_y, ang_z, p_cam_A, p_cam_B, N, case_proj):
    X = X.T
    X_Mo2w = find_Mo2w(c, koef_s, ang_x, ang_y, ang_z) @ X
    #print(ahf.find_Mo2w(c, koef_s, ang_x, ang_y, ang_z))
    X_Mw2c = find_Mw2c(p_cam_A, p_cam_B) @ X_Mo2w
    #print(X_Mw2c)
    X_Mproj = find_Mproj(case_proj) @ X_Mw2c
    X_Mproj[0] = X_Mproj[0]/X_Mproj[3]
    X_Mproj[1] = X_Mproj[1] / X_Mproj[3]
    X_Mproj[2] = X_Mproj[2] / X_Mproj[3]
    #print(X_Mproj)
    X_Mviewport = Mviewport(0, 0, N, X_Mproj[0], X_Mproj[1], X_Mproj[2])  # (x_w, y_w, z_n)
    return X_Mviewport

def new_Xn(X, c, koef_s, ang_x, ang_y, ang_z, p_cam_A, p_cam_B, N, case_proj):
    X = X.T
    X_Mo2w = np.linalg.inv(find_Mo2w(c, koef_s, ang_x, ang_y, ang_z).T) @ X
    X_Mw2c = np.linalg.inv(find_Mw2c(p_cam_A, p_cam_B).T) @ X_Mo2w
    X_Mproj = np.linalg.inv(find_Mproj(case_proj).T) @ X_Mw2c

    X_Mproj[0] = X_Mproj[0] / X_Mproj[3]
    X_Mproj[1] = X_Mproj[1] / X_Mproj[3]
    X_Mproj[2] = X_Mproj[2] / X_Mproj[3]

    X_Mviewport = Mviewport(0, 0, N, X_Mproj[0], X_Mproj[1], X_Mproj[2])  # (x_w, y_w, z_n)
    return X_Mviewport

def back_face_culling(p_cam_A, X1_xy, X2_xy, X3_xy):
    #sum = np.array(X1_n_xy) + np.array(X2_n_xy) + np.array(X3_n_xy)
    #norm_vect = sum / np.sqrt(sum[0]**2 + sum[1]**2 + sum[2]**2)
    v_vect = [X1_xy[0]-p_cam_A[0], X1_xy[1]-p_cam_A[1], X1_xy[2]-p_cam_A[2]]
    norm_vect = np.cross(np.array([X2_xy[0] - X1_xy[0], X2_xy[1] - X1_xy[1], X2_xy[2] - X1_xy[2]]),
                         np.array([X3_xy[0] - X2_xy[0], X3_xy[1] - X2_xy[1], X3_xy[2] - X2_xy[2]]))
    return np.dot(v_vect, norm_vect)

def barycentric_coords(x, y, v0, v1, v2):
    T = np.matrix([[v0[0], v1[0], v2[0]], [v0[1], v1[1], v2[1]], [1, 1, 1]])
    T_obr = np.linalg.inv(T)
    X = np.array([[x], [y], [1]])
    '''T = np.matrix([[v0[0] - v2[0], v1[0] - v2[0]],
                   [v0[1] - v2[1], v1[1] - v2[1]]])
    T_obr = np.linalg.inv(T)
    X = np.array([[x - v2[0]],
                  [y - v2[1]]])
    c = 1 - V[0] - V[1]
    V = np.array([V[0], V[1], c])'''
    V = T_obr @ X
    return V

def z_buf_f(z_buf, v0, v1, v2):
    x_min = min(v0[0], v1[0], v2[0])
    x_max = max(v0[0], v1[0], v2[0])
    y_min = min(v0[1], v1[1], v2[1])
    y_max = max(v0[1], v1[1], v2[1])
    for i in range(int(x_min), int(x_max)+1):
        for j in range(int(y_min), int(y_max)+1):
            V = barycentric_coords(i, j, v0, v1, v2)
            z_n = float(V[0] * v0[2] + V[1] * v1[2] + V[2] * v2[2])
            # print(z_buf)
            #print('z_n', z_n)
            # print(z_buf[i][ j])
            if V[0] >= 0 and V[1] >= 0 and V[2] >= 0:
                if z_n < z_buf[i, j]:
                    z_buf[i, j] = z_n
    return z_buf

def z_buf_im(z_buf, faces_v_vt_vn, c, koef_s, ang_x, ang_y, ang_z, p_cam_A, p_cam_B, N, case_proj):
    for face in faces_v_vt_vn:
        X1 = np.matrix([face[0][0][0], face[0][0][1], face[0][0][2], 1])
        X2 = np.matrix([face[0][1][0], face[0][1][1], face[0][1][2], 1])
        X3 = np.matrix([face[0][2][0], face[0][2][1], face[0][2][2], 1])

        X1_xy = new_X(X1, c, koef_s, ang_x, ang_y, ang_z, p_cam_A, p_cam_B, N, case_proj)
        X2_xy = new_X(X2, c, koef_s, ang_x, ang_y, ang_z, p_cam_A, p_cam_B, N, case_proj)
        X3_xy = new_X(X3, c, koef_s, ang_x, ang_y, ang_z, p_cam_A, p_cam_B, N, case_proj)

        z_buf = z_buf_f(z_buf, X1_xy, X2_xy, X3_xy)
    return z_buf

def rectangle(case, z_buf, canvas,texture_img, v0, v1, v2, v0_t, v1_t, v2_t):
    x_min = min(v0[0], v1[0], v2[0])
    x_max = max(v0[0], v1[0], v2[0])
    y_min = min(v0[1], v1[1], v2[1])
    y_max = max(v0[1], v1[1], v2[1])
    for i in range(int(x_min), int(x_max)+1):
        for j in range(int(y_min), int(y_max)+1):
            V = barycentric_coords(i, j, v0, v1, v2)
            z_n = float(V[0]*v0[2] + V[1]*v1[2] + V[2]*v2[2])
            if V[0] >= 0 and V[1] >= 0 and V[2] >= 0:
                if z_n == z_buf[i, j]:
                    print(z_n)
                    if case == 1:
                        draw_line(v0[0], v0[1], v1[0], v1[1], canvas, [255, 255, 255])
                        draw_line(v0[0], v0[1], v2[0], v2[1], canvas, [255, 255, 255])
                        draw_line(v2[0], v2[1], v1[0], v1[1], canvas, [255, 255, 255])
                    if case == 2:
                        '''sum = np.array(v1_n) + np.array(v2_n) + np.array(v3_n)
                        n = sum / np.sqrt(sum[0]**2 + sum[1]**2 + sum[2]**2)
                        n = np.cross(np.array([v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]]),
                                             np.array([v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]))
                        n_n = n/np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
                        p = np.array([i, j, v0[2]])
                        color = np.abs(np.dot(norm_vect, p)*255)
                        print('z_n', z_n)
                        l = np.array([0, 0, -1])
                        intensity = np.dot(l, n)
                        print(intensity)
                        color = [int(np.abs(255*intensity)), int(np.abs(255*intensity)), int(np.abs(255*intensity))]'''
                        color = np.array([np.abs(z_n*10)-2, np.abs(z_n*10)-2, np.abs(z_n*10)-2]) * 255
                        set_pixel(i, j, canvas, color)
                    if case == 3:
                        p = [[V[0] * v0_t[0] + V[1] * v1_t[0] + V[2] * v2_t[0]],
                             [V[0] * v0_t[1] + V[1] * v1_t[1] + V[2] * v2_t[1]]]
                        width, height = texture_img.size
                        i_texture, j_texture = p[0][0] * width, p[1][0] * height
                        pix = texture_img.load()
                        set_pixel(i, j, canvas, pix[i_texture, j_texture])
