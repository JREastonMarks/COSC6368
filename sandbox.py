import numpy as np
import math

def create_mask(matrix_size, dither_size, dither_x, dither_y):
    mask = np.zeros((matrix_size, matrix_size))
    offset = math.floor(dither_size / 2)

    dither_x_start = dither_x - offset
    dither_y_start = dither_y - offset

    dither_x_end = dither_x + offset
    dither_y_end = dither_y + offset

    for pos_x in range(0, matrix_size):
        for pos_y in range(0, matrix_size):
            if (dither_x_start <= pos_x <= dither_x_end) and (dither_y_start <= pos_y <= dither_y_end):
                weight = 1
            else:
                weight = 0.5 # Change to an offset 2^-x
            mask[pos_x, pos_y] = weight
    
    return mask

def create_masking(matrix_size, dither_size):
    offset = math.floor(dither_size / 2)
    start = offset
    end = matrix_size - offset
    mask = np.zeros((0,16))

    for dither_x in range(start, end):
        for dither_y in range(start, end):
            submask = create_mask(matrix_size, dither_size, dither_x, dither_y)
            submask.resize(1, matrix_size * matrix_size)
            
            mask = np.append(mask, submask, axis=0)

    return mask

results = create_masking(4, 3)

print(results)