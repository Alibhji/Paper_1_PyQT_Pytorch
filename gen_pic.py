
import matplotlib.pyplot as plt
import numpy as np
import random

def get_random_location(width, height, zoom=1.0):
    x = int(width * random.uniform(0.2, 0.8))
    y = int(height * random.uniform(0.2, 0.8))
    size = int(min(width, height) * random.uniform(0.06, 0.12) * zoom)
    return (x, y, size)

def add_squre(arr, x,y,size,fill=0):
    s=int(size/2)
    arr[x - s:x + s, y - s:y + s] = True
    if(fill):
        arr[x - s + fill:x + s - fill, y - s + fill:y + s - fill] = False
    return arr

def add_triangle(arr, x, y, size,fill=0):
    s =size
    # triangle = np.tril(np.ones((size, size), dtype=bool))
    # arr = triangle
    # arr[x - s:x - s + triangle.shape[0]-10, y - s+10:y - s + triangle.shape[1]-10] = False

    for i in (range(size)):
        # print(i)
        arr [y-i:y,x-i] = True
        # if(i and not i==size-1 and not i==size-2 and fill):
        #     arr[y - i+fill:y-fill, x - i] = False

    for i in (range(size-fill*2)):
        arr[y - i + fill:y-fill, x - i-fill] = False
        # arr[x - s:x + s, y - s:y + s, ch] = random.randrange(255)
    return arr

def add_circle(arr, x, y, size,fill=0):
    for i in range(size):
        r=int(size/2)
        for j in range(size):
            if ((i - r) ** 2 + (j - r) ** 2 <= (r) ** 2):
                arr[x + i, y + j] = True
            if ((i - r) ** 2 + (j - r) ** 2 <= (r-fill) ** 2) and fill:
                arr[x + i, y + j] = False
    return arr

def generate_img_and_mask(height, width,num=1,fill=0,circle=False,squre=False,triangle=False):
    shape = (height, width)
    arr = np.zeros(shape, dtype=np.uint8())
    mask_= np.zeros(shape, dtype=np.uint8())
    masks=[]
    annot={}

    if squre:
        squre_annot ={}
        for i in range(num):
            x, y, size = get_random_location(*shape, zoom=random.uniform(.2, 2))
            arr= add_squre(arr, x,y,size,fill)
            mask_ = add_squre(mask_, x, y, size, fill)
            squre_annot.update({'squre_{}'.format(i):[x,y,size]})
        masks.append(mask_)
        annot.update(squre_annot)

    mask_ = np.zeros(shape, dtype=np.uint8())

    if triangle:
        triangle_annot = {}
        for i in range(num):
            x, y, size = get_random_location(*shape, zoom=random.uniform(.2, 2))
            arr = add_triangle(arr, x, y, size,fill)
            mask_ = add_triangle(mask_, x, y, size, fill)
            triangle_annot.update({'triangle_{}'.format(i): [x, y, size]})
        masks.append(mask_)
        annot.update(triangle_annot)

    mask_ = np.zeros(shape, dtype=np.uint8())

    if circle:
        circle_annot = {}
        for i in range(num):
            x, y, size = get_random_location(*shape, zoom=random.uniform(.2, 2))
            arr = add_circle(arr, x, y, size, fill)
            mask_ = add_circle(mask_, x, y, size, fill)
            circle_annot.update({'circle_{}'.format(i): [x+int(size/2), y+int(size/2), size]})
        masks.append(mask_)
        annot.update(circle_annot)
    mask_ = np.zeros(shape, dtype=np.uint8())

    return np.reshape(arr,(1,height,width)), np.asarray(masks) ,annot


if __name__=='__main__':
    img=generate_img_and_mask(200,200)
    plt.imshow(img)
    plt.show()