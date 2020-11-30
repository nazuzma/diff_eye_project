import csv
import os
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from modules import coreimagemodule

root = os.getcwd()
print('root path: ', root)


def save_list_to_text_file(file_name, string_to_write):
    separate_lines_text = '\n'.join(string_to_write)
    text_file = open("{0}".format(file_name), "w")
    text_file.write(separate_lines_text)
    text_file.close()
    print("Saving file: '{0}' Successful".format(file_name))


def save_as_csv(file_name, hist1):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "Ring1"])
        for i in range(1, len(hist1)):
            writer.writerow([i, str(hist1[i]).replace('[', '').replace(']', '')])
            # , str(hist6[i]).replace('[', '').replace(']', ''),
        # str(hist7[i]).replace('[', '').replace(']', '')])
        print(file_name + ' saved!')


# array to read all the pixels of first ring and second ring
def get_inner_pixels_intensity(img_pointer, row, col):
    value_at_center = img_pointer[row, col]

    # if row == 178 and col == 287:
    #   print('hh')

    x_1 = 8
    y_1 = 8
    x_2 = -144
    y_2 = -144
    z_2 = -144

    first_ring = coreimagemodule.first_ring(img_pointer, row, col, x_1, y_1)
    # second_ring = coreimagemodule.second_ring(img_pointer, row, col, x_2, y_2, z_2)

    cent_int = (value_at_center - first_ring)  # + second_ring) #it wil be any result float
    result = (coreimagemodule.saturated_between_0_255_invert(
        cent_int))  # then befor making imqge this must be converted to int
    # if result>50:
    #   print('aaa')

    return result


# calculating Edge(inner to edge using edge) pixels

def get_edge_pixels_intensity(img_pointer, row, col):
    value_at_center = int(img_pointer[row, col])

    edge_ring = coreimagemodule.first_ring(img_pointer, row, col, 8, 8)

    cent_int = int(value_at_center - edge_ring)

    result = coreimagemodule.saturated_between_0_255_invert(cent_int)

    return 0


# create initial matrix according to height, width of image
def pixel_calculator(src):
    src_image = np.copy(src)
    height, width = src_image.shape  # defines grayscale(2D) image height and width
    print('The original image is of dimensions:', src_image.shape, ' data_type:', src_image.dtype, ' min_value:',
          np.amin(src_image), ' max_value:', np.amax(src_image))
    print(src_image)

    matrix = np.zeros(src_image.shape, src_image.dtype)  # calculated matrix made of equivalent dimension and data type
    matrix1 = np.zeros(src_image.shape, src_image.dtype)
    # matrix= np.copy(src_image)
    calc = 0

    # loop which goes through all required pixels
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            calc += 1
            # col-1 because we need to end at second last row
            if i == 1:  # Here we will calculate the first Row
                # print('img[{0},{1}]:{2}'.format(j, i, src_image[j, i]))
                new_intensity = get_edge_pixels_intensity(src_image, j, i)
                matrix[j, i] = new_intensity
                matrix1[j, i] = new_intensity
            elif i == width - 2:
                # Here we will calculate the last Row
                # print('img[{0},{1}]:{2}'.format(j, i, src_image[j, i]))
                new_intensity = get_edge_pixels_intensity(src_image, j, i)
                matrix[j, i] = new_intensity
                matrix1[j, i] = new_intensity
            elif i != 1 and i != width - 2 and j == 1:
                # Here we will calculate the first col
                # print('img[{0},{1}]:{2}'.format(j, i, src_image[j, i]))
                new_intensity = get_edge_pixels_intensity(src_image, j, i)
                matrix[j, i] = new_intensity
                matrix1[j, i] = new_intensity
            elif i != 1 and i != width - 2 and j == height - 2:
                # Here we will calculate the last col
                # print('img[{0},{1}]:{2}'.format(j, i, src_image[j, i]))
                new_intensity = get_edge_pixels_intensity(src_image, j, i)
                matrix[j, i] = new_intensity
                matrix1[j, i] = new_intensity
            elif i != 1 or i != width - 2 or j == height - 2 or j == 1:
                # Here we will calculate al remaining rows and cols
                # print('img[{0},{1}]:{2}'.format(j, i, src_image[j, i]))
                new_intensity = get_inner_pixels_intensity(src_image, j, i)
                matrix[j, i] = new_intensity
                matrix1[j, i] = new_intensity
                if matrix[j, i] == 255:
                    print('img[{0},{1}]:{2}'.format(j, i, matrix[j, i]))

    # loop which goes through all required pixels
    matrix3 = np.zeros(src_image.shape, src_image.dtype)
    print('The rescaled:', matrix3.shape, ' data_type:', matrix3.dtype, ' min_value:',
          np.amin(matrix3), ' max_value:', np.amax(matrix3))

    print('The calculated image is of dimensions:', matrix.shape, ' data_type:', matrix.dtype, ' min_value:',
          np.amin(matrix), ' max_value:', np.amax(matrix))
    print('The float calculated image is of dimensions:', matrix1.shape, ' data_type:', matrix1.dtype, ' min_value:',
          np.amin(matrix1), ' max_value:', np.amax(matrix1))

    return matrix, matrix1, matrix3


def main():
    root = os.getcwd()
    input_image_path = os.path.join(root, 'input_images', 'img_1.jpg')

    img = cv.imread(input_image_path)
    if img is None:
        print('Failed to load image:')
        sys.exit(1)
    src = coreimagemodule.color_to_grayscale(img)

    calculated_image, float_matrix, rescale_matrix = pixel_calculator(
        src)  # shows all intensity calculations by calling pixel_calculator function

    print(calculated_image)
    cv.imshow("Non-rescaled", float_matrix)  # displays calculated image
    cv.imwrite(os.path.join(root, 'Non-rescaled.png'), float_matrix)  # save calculated image

    # Histogram OpenCV
    float_matrix_hist = cv.calcHist([float_matrix], [0], None, [256], [0, 256])
    # cal_img = cv.calcHist([calculated_image], [0], None, [256], [0, 256])
    # plt.plot(cal_img)
    plt.plot(float_matrix_hist)
    plt.xlabel('Pixel Values')
    plt.ylabel('Number of Pixels')
    plt.xlim([0, 120])
    plt.yscale('log')
    plt.legend(('Non-rescaled Image'), loc='upper right')
    plt.savefig('output_images\\Non-rescaled.png')  # write at end to save the fig
    plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0


# Now we call the main function
main()
print('program end')
sys.exit(1)
