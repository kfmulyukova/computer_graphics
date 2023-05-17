import numpy as np
from PIL import Image as img

N = 1200
center = N/2, N/2
canvas = np.zeros((N, N, 3), dtype=np.uint8)
canvas[:, :] = [0, 0, 0]


def draw_pixel(x, y, array):
    x = int(x)
    y = int(y)
    p = 255 * (1 - (np.sqrt((x - N / 2) ** 2 + (y - N / 2) ** 2)) / N)
    color = [p, 0, 0]
    array[x, y] = color
    return array


def set_pixel(x, y):
    if x < 0:
        x = -round(x)
    else:
        x = round(x)
    if y < 0:
        y = -round(y)
    else:
        y = round(y)
    draw_pixel(x, y, canvas)


def coord(x, y, k):
    x = int((x + N/2) + x*k)
    y = int((y + N/2) - y*k)
    return y, x


def draw_line(x1, y1, x2, y2):
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

    set_pixel(x, y)

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
        set_pixel(x, y)

# список с вершинами граней
triangles = list()
with open('teapot.obj', 'r') as f:
    for line in f:
        if 'f' in line:
            a = line.split()
            v1, v2, v3 = a[1], a[2], a[3]
            triangles.append((v1, v2, v3))
with open('teapot.obj', 'r') as f:
    L = f.readlines()
    # проходим по списку с вершинами граней
    for t in triangles:
        # сдвигаем индексы вершин, чтобы они соответствовали верным строкам
        v1_index = int(t[0]) - 1
        v2_index = int(t[1]) - 1
        v3_index = int(t[2]) - 1
        # переходим к строке с нужным индексом
        v1 = L[v1_index]
        v2 = L[v2_index]
        v3 = L[v3_index]
        # разделяем вершины на координаты x, y, z
        v1 = v1.split()
        v2 = v2.split()
        v3 = v3.split()
        # удаляем первый элемент строки, т.е. букву v
        del v1[0]
        del v2[0]
        del v3[0]
        # меняем тип данных со стринг на флоат
        v1 = [float(i) for i in v1]
        v2 = [float(i) for i in v2]
        v3 = [float(i) for i in v3]

        x1, y1 = v1[0], v1[1]
        x2, y2 = v2[0], v2[1]
        x3, y3 = v3[0], v3[1]

        x1, y1 = coord(x1, y1, 172)
        x2, y2 = coord(x2, y2, 172)
        x3, y3 = coord(x3, y3, 172)

        draw_line(x1, y1, x2, y2)
        draw_line(x1, y1, x3, y3)
        draw_line(x3, y3, x2, y2)

im = img.fromarray(canvas)
im.save("teapot.png")
