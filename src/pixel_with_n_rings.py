import os
import sys

import cv2
import numpy as np

from modules import coreimagemodule

parameter_1 = 19.8  # 16 #13.3 #11.4 #10#8.9 #  10# #
parameter_2 = 66  # 64 #67 #76 #100#178 # #
parameter_3 = 165.5  # 192 #249 #381 #750#2667 # #
parameter_4 = 366  # 512 #842 #1778 #5000#35556 # #
parameter_5 = 762  # 1282 #2667 #8000 #33334 #31282#
parameter_6 = 1519  # 3077 #8000 #48000 #
parameter_7 = 2978  # 7179 #23334 #
parameter_8 = 5289  # 16410 #71112 #
parameter_9 = 12000  # 37894 #180000 #


def get_1st_ring(gray_img, row_number, col_number):
    current_pixel_intensity = gray_img[row_number][col_number]
    ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    result = (current_pixel_intensity - ring_intensity_sum_1)
    result = coreimagemodule.saturated_between_minus_255(result)
    return 0


def get_2nd_ring(gray_img, row_number, col_number):
    current_pixel_intensity = gray_img[row_number][col_number]
    ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    result = 0

    return (result)


def get_3rd_ring(gray_img, row_number, col_number):
    current_pixel_intensity = gray_img[row_number][col_number]
    ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    result = (current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3)
    result = coreimagemodule.saturated_between_minus_255(result)
    return 0


def get_4th_ring(gray_img, row_number, col_number):
    if row_number == 4 and col_number == 4:
        print()
    current_pixel_intensity = gray_img[row_number][col_number]
    ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    ring_intensity_sum_4 = pixel_neighbours(gray_img, row_number, col_number, 4) / parameter_4
    result = (
            current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3 - ring_intensity_sum_4)
    result = coreimagemodule.saturated_between_minus_255(result)
    return 0


def get_5th_ring(gray_img, row_number, col_number):
    current_pixel_intensity = gray_img[row_number][col_number]
    ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    ring_intensity_sum_4 = pixel_neighbours(gray_img, row_number, col_number, 4) / parameter_4
    ring_intensity_sum_5 = pixel_neighbours(gray_img, row_number, col_number, 5) / parameter_5
    result = (
            current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3 - ring_intensity_sum_4 - ring_intensity_sum_5)
    result = coreimagemodule.saturated_between_minus_255(result)
    return 0


def get_6th_ring(gray_img, row_number, col_number):
    current_pixel_intensity = gray_img[row_number][col_number]
    ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    ring_intensity_sum_4 = pixel_neighbours(gray_img, row_number, col_number, 4) / parameter_4
    ring_intensity_sum_5 = pixel_neighbours(gray_img, row_number, col_number, 5) / parameter_5
    ring_intensity_sum_6 = pixel_neighbours(gray_img, row_number, col_number, 6) / parameter_6
    result = current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3 - ring_intensity_sum_4 \
             - ring_intensity_sum_5 - ring_intensity_sum_6
    result = coreimagemodule.saturated_between_minus_255(result)
    return 0


def get_7th_ring(gray_img, row_number, col_number):
    current_pixel_intensity = gray_img[row_number][col_number]
    ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    ring_intensity_sum_4 = pixel_neighbours(gray_img, row_number, col_number, 4) / parameter_4
    ring_intensity_sum_5 = pixel_neighbours(gray_img, row_number, col_number, 5) / parameter_5
    ring_intensity_sum_6 = pixel_neighbours(gray_img, row_number, col_number, 6) / parameter_6
    ring_intensity_sum_7 = pixel_neighbours(gray_img, row_number, col_number, 7) / parameter_7
    result = int(
        current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3 - ring_intensity_sum_4 - ring_intensity_sum_5 - ring_intensity_sum_6 - ring_intensity_sum_7)
    result = coreimagemodule.saturated_between_minus_255(result)
    return 0


def get_8th_ring(gray_img, row_number, col_number):
    current_pixel_intensity = gray_img[row_number][col_number]
    ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    ring_intensity_sum_4 = pixel_neighbours(gray_img, row_number, col_number, 4) / parameter_4
    ring_intensity_sum_5 = pixel_neighbours(gray_img, row_number, col_number, 5) / parameter_5
    ring_intensity_sum_6 = pixel_neighbours(gray_img, row_number, col_number, 6) / parameter_6
    ring_intensity_sum_7 = pixel_neighbours(gray_img, row_number, col_number, 7) / parameter_7
    ring_intensity_sum_8 = pixel_neighbours(gray_img, row_number, col_number, 8) / parameter_8
    result = int(
        current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3 - ring_intensity_sum_4 - ring_intensity_sum_5 - ring_intensity_sum_6 - ring_intensity_sum_7 - ring_intensity_sum_8)
    result = coreimagemodule.saturated_between_minus_255(result)
    return 0


def get_9th_ring(gray_img, row_number, col_number):
    current_pixel_intensity = gray_img[row_number][col_number]
    ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    ring_intensity_sum_4 = pixel_neighbours(gray_img, row_number, col_number, 4) / parameter_4
    ring_intensity_sum_5 = pixel_neighbours(gray_img, row_number, col_number, 5) / parameter_5
    ring_intensity_sum_6 = pixel_neighbours(gray_img, row_number, col_number, 6) / parameter_6
    ring_intensity_sum_7 = pixel_neighbours(gray_img, row_number, col_number, 7) / parameter_7
    ring_intensity_sum_8 = pixel_neighbours(gray_img, row_number, col_number, 8) / parameter_8
    ring_intensity_sum_9 = pixel_neighbours(gray_img, row_number, col_number, 9) / parameter_9
    result = current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3 - ring_intensity_sum_4 \
             - ring_intensity_sum_5 - ring_intensity_sum_6 - ring_intensity_sum_7 - ring_intensity_sum_8 - ring_intensity_sum_9
    result = coreimagemodule.saturated_between_minus_255(result)
    return result


def pixel_neighbours(img, row_number, col_number, ring_number):
    # if ring_number is not 9:
    #    print(ring_number, 'do whatever you want to do')
    rows, cols = img.shape
    minimum_row = row_number - ring_number if row_number - ring_number >= 0 else 0
    maximum_row = row_number + ring_number if row_number + ring_number < rows else row_number

    minimum_col = col_number - ring_number if col_number - ring_number >= 0 else 0
    maximum_col = col_number + ring_number if col_number + ring_number < cols else col_number

    neighbours = []
    neighbours_intensity = 0

    for x in range(minimum_row, maximum_row + 1):
        for y in range(minimum_col, maximum_col + 1):
            if (y == minimum_col or y == maximum_col) or (x == minimum_row or x == maximum_row):
                # only add those pixels which are in that ring
                neighbours.append([x, y])  # here we can also get img[x,y] intensity, x as row, y as col
                neighbours_intensity = neighbours_intensity + img[x, y]

    return neighbours_intensity


def pixel_calculator(src):
    src_image = np.copy(src)
    height, width = src_image.shape

    new_img = np.zeros(src_image.shape, src_image.dtype)  # calculated matrix made of equivalent dimension and data type
    cal_new_img = np.zeros(src_image.shape, )  # created matrix for non-rescaled image

    total_rows, total_cols = src.shape  # defines grayscale(2D) image height and width
    calc = 0
    # loop through whole image except 1st row,col,last row and col. they are set to zero
    for row_number in range(1, total_rows - 1):
        for col_number in range(1, total_cols - 1):
            # print(row_number, col_number)

            if row_number == 1 or col_number == 1 or row_number == total_rows - 2 or col_number == total_cols - 2:
                new_intensity = get_1st_ring(src_image, row_number, col_number)
                new_img[row_number][col_number] = new_intensity
                cal_new_img[row_number][col_number] = new_intensity
            elif row_number == 2 or col_number == 2 or row_number == total_rows - 3 or col_number == total_cols - 3:
                new_intensity = get_2nd_ring(src_image, row_number, col_number)
                new_img[row_number][col_number] = new_intensity
                cal_new_img[row_number][col_number] = new_intensity
            elif row_number == 3 or col_number == 3 or row_number == total_rows - 4 or col_number == total_cols - 4:
                new_intensity = get_3rd_ring(src_image, row_number, col_number)
                new_img[row_number][col_number] = new_intensity
                cal_new_img[row_number][col_number] = new_intensity
            elif row_number == 4 or col_number == 4 or row_number == total_rows - 5 or col_number == total_cols - 5:
                new_intensity = get_4th_ring(src_image, row_number, col_number)
                new_img[row_number][col_number] = new_intensity
                cal_new_img[row_number][col_number] = new_intensity
            elif row_number == 5 or col_number == 5 or row_number == total_rows - 6 or col_number == total_cols - 6:
                new_intensity = get_5th_ring(new_img, row_number, col_number)
                new_img[row_number][col_number] = new_intensity
                cal_new_img[row_number][col_number] = new_intensity
            elif row_number == 6 or col_number == 6 or row_number == total_rows - 7 or col_number == total_cols - 7:
                new_intensity = get_6th_ring(new_img, row_number, col_number)
                new_img[row_number][col_number] = new_intensity
                cal_new_img[row_number][col_number] = new_intensity
            elif row_number == 7 or col_number == 7 or row_number == total_rows - 8 or col_number == total_cols - 8:
                new_intensity = get_7th_ring(new_img, row_number, col_number)
                new_img[row_number][col_number] = new_intensity
                cal_new_img[row_number][col_number] = new_intensity
            elif row_number == 8 or col_number == 8 or row_number == total_rows - 9 or col_number == total_cols - 9:
                new_intensity = get_8th_ring(new_img, row_number, col_number)
                new_img[row_number][col_number] = new_intensity
                cal_new_img[row_number][col_number] = new_intensity
            elif row_number == 9 or col_number == 9 or row_number == total_rows - 10 or col_number == total_cols - 10:
                new_intensity = get_9th_ring(new_img, row_number, col_number)
                new_img[row_number][col_number] = new_intensity
                cal_new_img[row_number][col_number] = new_intensity

            else:
                new_intensity = get_9th_ring(src_image, row_number, col_number)
                new_img[row_number][col_number] = new_intensity
                cal_new_img[row_number][col_number] = new_intensity

    print(cal_new_img)
    cal_min = np.amin(cal_new_img)
    cal_max = np.amax(cal_new_img)
    print('The calculated image is of dimensions:', cal_new_img.shape, ' data_type:', cal_new_img.dtype,
          ' min_value_cal_img:', cal_min, ' max_value_cal_img:', cal_max)

    rescale_img = np.zeros(cal_new_img.shape,
                           src_image.dtype)  # calculated matrix made of equivalent dimension and data type
    res_height, res_width = new_img.shape  # defines grayscale(2D) image height and width

    for j1 in range(0, res_height):
        for i1 in range(0, res_width):
            rescale_img[j1, i1] = coreimagemodule.rescale_non_inversion(cal_new_img, j1, i1, cal_min, cal_max)
            # rescale_to_min_max
            # rescale_non_inversion

    print('new_min:', np.amin(rescale_img), ' new_max:', np.amax(rescale_img))

    return new_img, cal_new_img, rescale_img


def main():
    root = os.getcwd()
    input_image_path = os.path.join(root, 'input_images', 'img_1.jpg')
    input_image = cv2.imread(input_image_path)

    if input_image is None:
        print('Failed to load image:')
        sys.exit(1)

    src = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    new_img, calc_new_img, rescale_img = pixel_calculator(
        src)  # shows all intensity calculations by calling pixel_calculator function

    # print(calculated_image)
    cv2.imshow("Calculated", rescale_img)  # displays calculated image
    # cv2.imwrite(os.path.join(root, 'test.png'), rescale_img)  # save calculated image
    cv2.imwrite(os.path.join(root, 'inv_img2_ring_9C.png'), rescale_img)  # save calculated image

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


# Now we call the main function
main()
print('program end')
sys.exit(1)
