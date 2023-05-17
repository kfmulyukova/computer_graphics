import random
import numpy as np
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


def draw_line(x1, y1, x2, y2, color):
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

    set_pixel(x, y, color)

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
        set_pixel(x, y, color)


def draw_square(a, color):
    x1 = N / 6
    y1 = N / 6
    x2 = x1 + a
    y2 = x1 + a

    draw_line(x1, y2, x1, y1, color)
    draw_line(x2, y2, x2, y1, color)
    draw_line(x1, y2, x2, y2, color)
    draw_line(x1, y1, x2, y1, color)


def fill_circle(circle_color):
    for x in range(center - r, center + r):
        for y in range(center - r, center + r):
            if np.sqrt((x - center)**2 + (y - center)**2) < r:
                canvas[x, y] = circle_color


def draw_circle(color, x, y, radius):
    _x = 0
    _y = radius
    delta = (2 - 2 * radius)
    while _y >= 0:
        set_pixel(x + _x, y + _y, color)
        set_pixel(x + _x, y - _y, color)
        set_pixel(x - _x, y - _y, color)
        set_pixel(x - _x, y + _y, color)
        gap = 2 * (delta + _y) - 1
        if delta < 0 and gap <= 0:
            _x = _x+1
            delta += 2 * _x + 1
            continue
        if delta > 0 and gap > 0:
            _y = _y - 1
            delta -= 2 * _y + 1
            continue
        _x = _x+1
        delta += 2 * (_x - _y)
        _y = _y - 1
    fill_circle(color)


def draw_triangle_by_points(x_new, color):
    draw_line(x_new[0][0], x_new[0][1], x_new[1][0], x_new[1][1], color)
    draw_line(x_new[0][0], x_new[0][1], x_new[2][0], x_new[2][1], color)
    draw_line(x_new[1][0], x_new[1][1], x_new[2][0], x_new[2][1], color)


def draw_triangle(tr_center, r_t, n, color):
    p = list()
    z = 0
    angle = 360/n
    for i in range(3):
        _p = tr_center[0]+(np.round(np.cos(z/180*np.pi)*r_t))
        p.append(_p)
        _p = tr_center[1]+(np.round(np.sin(z/180*np.pi)*r_t))
        p.append(_p)
        z = z + angle
    draw_line(p[0], p[1], p[2], p[3], color)
    draw_line(p[0], p[1], p[4], p[5], color)
    draw_line(p[2], p[3], p[4], p[5], color)


def add(point_1, point_2):
    new_point = (point_1[0] + point_2[0], point_1[1] + point_2[1])
    return new_point


def check_coords(rot_coord, velocity, r, triangle_color, square_color, circle_color):
    triangle = rot_coord
    for t in triangle:
        if t[0] <= N/6 + 10 or t[0] >= N - N/6 - 10:
            velocity[0] = -velocity[0]
            triangle_color, square_color = square_color, triangle_color
            draw_square(800, square_color)
        if ((t[0] - center) ** 2 + (t[1] - center) ** 2) <= (r + 10) ** 2:
            velocity[0] = -velocity[0]
            velocity[1] = -velocity[1]
            triangle_color, circle_color = circle_color, triangle_color
            draw_circle(circle_color, center, center, r)
        if t[1] <= N/6 + 10 or t[1] >= N - N/6 - 10:
            velocity[1] = -velocity[1]
            triangle_color, square_color = square_color, triangle_color
            draw_square(800, square_color)
    return (triangle_color, square_color, circle_color)


def triangle_start():
    x_c = random.randint(N//6+r_t*2 + 1, N - N//6-r_t*2 - 1)
    if (x_c <= center - r - r_t*2 - 1) or (x_c >= center + r + r_t*2 + 1):
        y_c_start = random.randint(N//6+r_t*2 + 1, N - N//6-r_t*2 - 1)
    elif center - r - r_t*2 <= x_c <= center + r + r_t*2:
        y_c1 = random.randint(N // 6 + r_t*2 + 1, center - r - r_t*2 - 1)
        y_c2 = random.randint(center + r + r_t*2 + 1, N - N//6 - r_t*2 - 1)
        y_c_start = random.choice([y_c1, y_c2])
    tr_center_start = (x_c, y_c_start)
    return tr_center_start


def rotation(ang, tr_center, r_t, angles):
    rot_coord = list()
    triangle = angles
    for t in triangle:
        rot_x = tr_center[0] + r_t * math.cos(t + ang)
        rot_y = tr_center[1] + r_t * math.sin(t + ang)
        rot_coord.append((rot_x, rot_y))
    return rot_coord


N = 1200
center = N//2
r = 200
r_t = 30
canvas = np.zeros((N, N, 3), dtype=np.uint8)
canvas[:, :] = [0, 0, 0]
images = []

tr_center = triangle_start()
draw_square(800, [57, 255, 20])
draw_circle([255, 7, 58], center, center, r)
draw_triangle(tr_center, r_t, 3, [255, 255, 255])
im = img.fromarray(canvas)
images.append(im)
draw_triangle(tr_center, r_t, 3, [0, 0, 0])

velocity = list()
a = random.choice([-5, 5])
while a == 0:
    a = random.choice([-5, 5])
b = random.choice([-5, 5])
while b == 0:
    b = random.choice([-5, 5])
velocity.append(a)
velocity.append(b)

angles = [0, 2*np.pi/3, 4*np.pi/3]
new_tr_center = add(tr_center, velocity)
triangle_color = [255, 255, 255]
square_color = [57, 255, 20]
circle_color = [255, 7, 58]
# r_t1 - постепенно изменяющийся радиус
r_t1 = r_t
# k - коэф-т, если к = 1 - увеличивается, к = -1 - уменьшаетсяб сначала увел, тк за нечет поворот увеличивается
k = 1
for i in range(150):
    # ang - угол поворота треугольника
    ang = np.pi / 20
    # если уменьшается и еще не дошел до начального размера
    if k == -1 and r_t1 >= r_t:
        r_t1 -= r_t/20
    # если уменьшается и дошел до начального размера, меняем коэф-т
    if k == -1 and r_t1 == r_t:
        k = 1
    # если увеличивается и еще не стал в 2 раза больше
    if k == 1 and r_t1 <= r_t * 2:
        r_t1 += r_t/20
    # если увеличивается и стал больше в 2 раза, меняем коэф-т
    if k == 1 and r_t1 == r_t * 2:
        k = -1
    new_tr_center = add(new_tr_center, velocity)
    rot_coord = rotation(ang, new_tr_center, r_t1, angles)
    triangle_color,square_color,circle_color=check_coords(rot_coord,velocity,r,triangle_color,square_color,circle_color)

    draw_triangle_by_points(rot_coord, triangle_color)
    im = img.fromarray(canvas)
    images.append(im)

    draw_triangle_by_points(rot_coord, [0, 0, 0])
    angles = [angles[0] + ang, angles[1] + ang, angles[2] + ang]

images[0].save('rotation.gif', save_all=True, append_images=images[1:], optimize=False, duration=70, loop=0)
