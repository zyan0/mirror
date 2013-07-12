from __future__ import print_function
import Image
import math
import random

def map_function(x):
    if 0 <= x and x <= 63:
        return 0
    elif 64 <= x and x <= 127:
        return 1
    elif 128 <= x and x <= 191:
        return 2
    else:
        return 3

def calc_vect(im):
    vect = [0] * 64
    cnt = 0
    for pixel in im.getdata():
        if random.random() > 0.2:
            continue
        cnt += 1
        rgb = map(map_function, pixel)
        data = map(lambda x: str(x), rgb)
        data = ''.join(data)
        no = int(data, 4)
        vect[no] += 1
    return vect

def mult(vect_1, vect_2):
    result = 0
    for i, j in zip(vect_1, vect_2):
        result += i * j
    return result

def calc_dist(vect_1, vect_2):
    return mult(vect_1, vect_2) / math.sqrt(mult(vect_1, vect_1)) / math.sqrt(mult(vect_2, vect_2))

def calc_image_dist(x, y):
    id_vect_1 = calc_vect(x)
    id_vect_2 = calc_vect(y)
    return calc_dist(id_vect_1, id_vect_2)

def match(image1, image2):
    return calc_image_dist( Image.open(image1), Image.open(image2) )

if __name__ == '__main__':
    pass