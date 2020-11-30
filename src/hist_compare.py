import os
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from modules import coreimagemodule

root = os.getcwd()


# array to read all the pixels of first ring and second ring
def get_inner_pixels_intensity(img_pointer, row, col):
    value_at_center = int(img_pointer[row, col])
    x_1 = 9
    y_1 = 9
    x_2 = 144
    y_2 = 144
    z_2 = 144

    first_ring = coreimagemodule.first_ring(img_pointer, row, col, x_1, y_1)
    second_ring = coreimagemodule.second_ring(img_pointer, row, col, x_2, y_2, z_2)

    cent_int = (value_at_center - first_ring - second_ring)
    result = coreimagemodule.saturated_between_0_255_invert(cent_int)
    return result


# calculating Edge(inner to edge using edge) pixels

def get_edge_pixels_intensity(img_pointer, row, col):
    value_at_center = (img_pointer[row, col])
    x_1 = 9
    y_1 = 9
    edge_ring = coreimagemodule.first_ring(img_pointer, row, col, x_1, y_1)
    cent_int = (value_at_center - edge_ring)
    result = coreimagemodule.saturated_between_0_255_invert(cent_int)
    return 0


def get_inner_pixels_intensity1(img_pointer, row, col):
    value_at_center = (img_pointer[row, col])
    x_1 = 9
    y_1 = 9
    x_2 = 144
    y_2 = 144
    z_2 = 144

    first_ring = coreimagemodule.first_ring(img_pointer, row, col, x_1, y_1)
    second_ring = coreimagemodule.second_ring(img_pointer, row, col, x_2, y_2, z_2)

    cent_int = (value_at_center - first_ring - second_ring)
    result = coreimagemodule.saturated_between_minus_255(cent_int)
    return result


# calculating Edge(inner to edge using edge) pixels

def get_edge_pixels_intensity1(img_pointer, row, col):
    value_at_center = (img_pointer[row, col])
    x_1 = 9
    y_1 = 9
    edge_ring = coreimagemodule.first_ring(img_pointer, row, col, x_1, y_1)
    cent_int = (value_at_center - edge_ring)
    result = coreimagemodule.saturated_between_minus_255(cent_int)
    return 0


# create initial matrix according to height, width of image
def pixel_calculator(src):
    src_image = np.copy(src)
    height, width = src_image.shape  # defines grayscale(2D) image height and width
    # print(src_image)
    matrix = np.zeros(src_image.shape)
    # calculated matrix made of equivalent dimension and data type
    matrix1 = np.zeros(src_image.shape)  # calculated matrix made of equivalent dimension and data type

    calc = 0
    # loop which goes through all required pixels
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            calc += 1
            # col-1 because we need to end at second last row
            if i == 1:  # Here we will calculate the first Row
                # print('img[{0},{1}]:{2}'.format(j, i, src_image[j, i]))
                matrix[j, i] = get_edge_pixels_intensity(src_image, j, i)
                matrix1[j, i] = get_edge_pixels_intensity1(src_image, j, i)
            elif i == width - 2:
                # Here we will calculate the last Row
                # print('img[{0},{1}]:{2}'.format(j, i, src_image[j, i]))
                matrix[j, i] = get_edge_pixels_intensity(src_image, j, i)
                matrix1[j, i] = get_edge_pixels_intensity1(src_image, j, i)
            elif i != 1 and i != width - 2 and j == 1:
                # Here we will calculate the first col
                # print('img[{0},{1}]:{2}'.format(j, i, src_image[j, i]))
                matrix[j, i] = get_edge_pixels_intensity(src_image, j, i)
                matrix1[j, i] = get_edge_pixels_intensity1(src_image, j, i)
            elif i != 1 and i != width - 2 and j == height - 2:
                # Here we will calculate the last col
                # print('img[{0},{1}]:{2}'.format(j, i, src_image[j, i]))
                matrix[j, i] = get_edge_pixels_intensity(src_image, j, i)
                matrix1[j, i] = get_edge_pixels_intensity1(src_image, j, i)
            elif i != 1 or i != width - 2 or j == height - 2 or j == 1:
                # Here we will calculate al remaining rows and cols
                # print('img[{0},{1}]:{2}'.format(j, i, src_image[j, i]))
                matrix[j, i] = get_inner_pixels_intensity(src_image, j, i)
                matrix1[j, i] = get_inner_pixels_intensity1(src_image, j, i)

    print(matrix)
    print(matrix1)

    max = coreimagemodule.get_max_intensity(matrix)
    min = coreimagemodule.get_min_intensity(matrix)
    print('min is:', min, ' max is:', max)

    print('Non-inv_min:', np.amin(matrix1), ' Non-inv_max:', np.amax(matrix1))

    fig, axs = plt.subplots(nrows=2, figsize=(10, 10))
    axs[0].set_title('Inverted Non-rescaled Image Ring 2(-9, -144)')
    axs[0].set_ylabel('Number of Pixels')
    axs[0].set_yscale('log')
    axs[0].set_xlim([-255, 255])
    hist_original = axs[0].hist(matrix.ravel(), 256, [0, 256])

    axs[1].set_title('Non-Inverted Non-rescaled Image')
    axs[1].set_xlabel('Pixel Values')
    axs[1].set_ylabel('Number of Pixels')
    axs[1].set_yscale('log')
    axs[1].set_xlim([-255, 255])
    hist_cal = axs[1].hist(matrix1.ravel(), 256, [-256, 256])
    plt.savefig('Inv_non-inv_-9_-144.png')
    plt.show()


def main():
    root_path = root + '\\Images\\New_image\\Circle_new.png'
    img = cv.imread(root_path)
    if img is None:
        print('Failed to load image:')
        sys.exit(1)
    src = coreimagemodule.color_to_grayscale(img)

    pixel_calculator(src)  # shows all intensity calculations by calling pixel_calculator function

    cv.waitKey(0)

    cv.destroyAllWindows()
    return 0


# Now we call the main function
main()
