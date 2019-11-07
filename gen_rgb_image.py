import numpy as np
import random

def get_random_location(width, height,depth, zoom=1.0):
    x = int(width * random.uniform(0.1, 0.9))
    y = int(height * random.uniform(0.1, 0.9))

    size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)

    return (x, y, size)


def add_squre(arr, x,y,size):
    s=int(size/2)
    val=np.random.randint(0,255,size=(3))
    print(val)
    print('*'*20)
    for ch in range(2):
        arr[x - s:x + s, y - s:y + s, ch] = val[ch]
        # arr[x - s:x + s, y - s:y + s, ch] = random.randrange(255)

    return arr


def add_squre(arr, x,y,size):
    s=int(size/2)
    val=np.random.randint(0,255,size=(3))
    for ch in range(2):
        arr[x - s:x + s, y - s:y + s, ch] = val[ch]
        # arr[x - s:x + s, y - s:y + s, ch] = random.randrange(255)

    return arr

def add_triangle(arr, x, y, size):
    s = int(size / 2)
    triangle = np.tril(np.ones((size, size), dtype=bool))
    val=np.random.randint(0,255,size=(3))
    for ch in range(2):
        arr[x - s:x - s+triangle.shape[0], y - s :y - s + triangle.shape[1], ch] = val[ch] * triangle
        # arr[x - s:x + s, y - s:y + s, ch] = random.randrange(255)


    return arr


def generate_img_and_mask(height, width,depth):
    shape = (height, width,depth)

    # triangle_location = get_random_location(*shape)
    # circle_location1 = get_random_location(*shape, zoom=0.7)
    # circle_location2 = get_random_location(*shape, zoom=0.5)
    # mesh_location = get_random_location(*shape)
    # square_location = get_random_location(*shape, zoom=0.8)
    # plus_location = get_random_location(*shape, zoom=1.2)

    # Create input image
    arr = np.zeros(shape, dtype=np.uint8())
    for i in range(100):
        x,y,sizd=get_random_location(*shape)
        # arr= add_squre (arr, x,y,sizd)
        arr = add_triangle(arr, x, y, sizd)

    # arr = add_triangle(arr, *triangle_location)
    # arr = add_circle(arr, *circle_location1)
    # arr = add_circle(arr, *circle_location2, fill=True)
    # arr = add_mesh_square(arr, *mesh_location)
    # arr = add_filled_square(arr, *square_location)
    # arr = add_plus(arr, *plus_location)
    # arr = np.reshape(arr, (1, height, width)).astype(np.float32)


    return arr


import matplotlib.pyplot as plt
img=generate_img_and_mask(100,100,3)
plt.imshow(img)
plt.show()