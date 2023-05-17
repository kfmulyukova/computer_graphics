import matplotlib.pyplot as plt
from matplotlib.patches import *
from PIL import Image as img
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'


def draw_pixel(x, y, array, color):
    x = int(x)
    y = int(y)
    array[x, y] = color
    return array


def set_pixel(x, y, color):
    if x < 0:
        x = -round(x)
    else:
        x = round(x)
    if y < 0:
        y = -round(y)
    else:
        y = round(y)
    draw_pixel(x, y, canvas, color)


def bezier(t_values, points, color):
    curve = list()
    p0, p1, p2 = points[0], points[1], points[2]
    for t in t_values:
        x_t = (1 - t) ** 2 * p0[0] + 2 * t * (1 - t) * p1[0] + t ** 2 * p2[0]
        y_t = (1 - t) ** 2 * p0[1] + 2 * t * (1 - t) * p1[1] + t ** 2 * p2[1]
        set_pixel(x_t, y_t, color)


def f_control_points(K, center, r, l, q, up_or_down):
    ang = 2*np.pi/K
    C = 2*np.pi
    f_control_points = list()
    k = 1
    if q == 0:
        while C >= 0:
            if k == 1:
                x = center + (1 + l) * r * np.cos(C)
                y = center + (1 + l) * r * np.sin(C)
            elif k == -1:
                x = center + (1 - l) * r * np.cos(C)
                y = center + (1 - l) * r * np.sin(C)
            f_control_points.append((x, y))
            C -= ang
            k = -k
    elif q == 1 and up_or_down == 0:
        while C >= 0:
            if k == 1:
                x = center + (1 - l) * r * np.cos(C)
                y = center + (1 - l) * r * np.sin(C)
            elif k == -1:
                x = center + (1 + l) * r * np.cos(C)
                y = center + (1 + l) * r * np.sin(C)
            f_control_points.append((x, y))
            C -= ang
            k = -k
    elif q == 1 and up_or_down == 1:
        while C >= 0:
            if k == 1:
                x = center + (1 + l) * r * np.cos(C)
                y = center + (1 + l) * r * np.sin(C)
            elif k == -1:
                x = center + (1 - l) * r * np.cos(C)
                y = center + (1 - l) * r * np.sin(C)
            f_control_points.append((x, y))
            C -= ang
            k = -k
    elif q == -1 and up_or_down == 1:
        while C >= 0:
            if k == 1:
                x = center + (1 + l) * r * np.cos(C)
                y = center + (1 + l) * r * np.sin(C)
            elif k == -1:
                x = center + (1 - l) * r * np.cos(C)
                y = center + (1 - l) * r * np.sin(C)
            f_control_points.append((x, y))
            C -= ang
            k = -k
    elif q == -1 and up_or_down == 0:
        while C >= 0:
            if k == 1:
                x = center + (1 - l) * r * np.cos(C)
                y = center + (1 - l) * r * np.sin(C)
            elif k == -1:
                x = center + (1 + l) * r * np.cos(C)
                y = center + (1 + l) * r * np.sin(C)
            f_control_points.append((x, y))
            C -= ang
            k = -k
    return f_control_points


N = 1200
K = 48
center = N/2
r = 300
color = [255, 255, 255]
canvas = np.zeros((N, N, 3), dtype=np.uint8)
canvas[:, :] = [0, 0, 0]
images = []

t_values = np.arange(0, 1, 0.005)
points = list()
#
l = 0
# q = 0 - начальное положение, q = 1 - направление движения от центра окружности, q = -1 - направление дижения к центру
q = 0
# up_or_down  1 - движение за пределами радиуса, 0 - движение в пределах радиуса
up_or_down = 1
# положение контрольных точек на начальной окружности
control_points = f_control_points(K, center, r, l, q, up_or_down)
for i in range(140):
    # индекс контрольной точки
    j = 0
    canvas[:, :] = [0, 0, 0]
    while j < K:
        new_control_points = f_control_points(K, center, r, l, q, up_or_down)
        # b - контрольная точка, a и c - опорные точки
        b = new_control_points[j]
        if j == (K - 1):
            c = ((control_points[0][0] + control_points[j][0]) / 2, (control_points[0][1] + control_points[j][1])/2)
        else:
            c = ((control_points[j + 1][0] + control_points[j][0])/2, (control_points[j+1][1] + control_points[j][1])/2)
        if j == 0:
            a = ((control_points[K-1][0]+control_points[j][0])/2, (control_points[K-1][1]+control_points[j][1])/2)
        else:
            a = ((control_points[j-1][0]+control_points[j][0])/2, (control_points[j-1][1]+control_points[j][1])/2)
        points = (a, b, c)
        bezier(t_values, points, color)
        j += 1
    im = img.fromarray(canvas)
    images.append(im)

    if q == 0 and l <= 0.5:
        l += 0.5/50
    if q == 0 and l > 0.5:
        l -= 0.5 / 50
        q = -1
    if q == -1:
        if up_or_down == 1:
            if l >= 0.5 / 50:
                l -= 0.5 / 50
            if l < 0.5/50:
                l += 0.5/50
                up_or_down = 0
        if up_or_down == 0:
            if l <= 0.5:
                l += 0.5 / 50
            if l > 0.5:
                l -= 0.5/50
                q = 1
    if q == 1:
        if up_or_down == 0:
            if l >= 0.5:
                l -= 0.5 / 50
            if l < 0.5/50:
                l += 0.5/50
                up_or_down = 1
        if up_or_down == 1:
            if l <= 0.5/50:
                l += 0.5 / 50
            if l > 0.5:
                l -= 0.5/50
                q = -1

images[0].save('bezier.gif', save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)
