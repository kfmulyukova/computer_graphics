import numpy as np
from collections import Counter

# множество ребер
edges = set()
with open('teapot.obj', 'r') as f:
    for line in f:
        if 'f' in line:
            a = line.split()
            # попарно добавляем вершины из граней в множество ребер, добавляя в пару сначала вершину с меньшим индексом,
            # чтобы сет убрал повторяющиеся ребра
            edges.add((min(a[1], a[2]), max(a[1], a[2])))
            edges.add((min(a[2], a[3]), max(a[2], a[3])))
            edges.add((min(a[1], a[3]), max(a[1], a[3])))

with open('teapot.obj', 'r') as f:
    L = f.readlines()
    edges_sum = 0
    for e in edges:
        v1_index = int(e[0]) - 1
        v2_index = int(e[1]) - 1
        # переходим к строке с нужным индексом
        v1 = L[v1_index]
        v2 = L[v2_index]

        v1 = v1.split()
        v2 = v2.split()

        del v1[0]
        del v2[0]

        v1 = [float(i) for i in v1]
        v2 = [float(i) for i in v2]
        edges_sum += np.sqrt(np.power((v2[0] - v1[0]), 2) + np.power((v2[1] - v1[1]), 2) + np.power((v2[2] - v1[2]), 2))
print('Суммарная длина всех ребер: ' + str(edges_sum))

# список из индексов вершин
vertexes_index = list()
with open('teapot.obj', 'r') as f:
    for line in f:
        if 'f' in line:
            b = line.split()
            del b[0]
            b = [int(i) for i in b]
            vertexes_index.append(b[0])
            vertexes_index.append(b[1])
            vertexes_index.append(b[2])

counts = Counter(vertexes_index)
# макс число граней
max_count = counts.most_common(1)[0][1]
out = [value for value, count in counts.most_common() if count == max_count]

print('Вершины, принадлежащие максимальному количеству граней: ' + str(out))
print('Количество граней: ' + str(max_count))
# через L обращение к строке по индексу со сдвигом -1
vert1 = L[out[0] - 1]
vert2 = L[out[1] - 1]
print('Координаты первой вершины: ' + vert1)
print('Координаты второй вершины: ' + vert2)
