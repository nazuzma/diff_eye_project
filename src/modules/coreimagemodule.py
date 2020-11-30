import cv2 as cv
import numpy as np


# verification of Gray scale image, if it is gray scale then it returns true otherwise false
def is_grayscale(source_image):
    return len(source_image.shape) < 3


# conversion of color to gray scale
def color_to_grayscale(src_image):
    if is_grayscale(src_image):  # if image is already graysclae no need to convert, just return
        return src_image
    else:
        return cv.cvtColor(src_image,
                           cv.COLOR_BGR2GRAY)  # using opencv builtin function to convert color image to graysclae


# Saturate between 0 and 255, if minus then multiply with - to make +, limit max to 255 by below equation
def saturated_between_0_255(sum_value):
    if sum_value < 0:
        return sum_value * 0
    elif sum_value > 255:
        return 255 - (sum_value - 255)
    else:
        return sum_value


def saturated_between_0_255_invert(sum_value):
    if sum_value < 0:
        if sum_value < -255:
            print(sum_value)
        return sum_value * -1
    elif sum_value > 255:
        return 255 - (sum_value - 255)
    else:
        return sum_value


# Saturate between minus and 255, limit max to 255 by below equation
def saturated_between_minus_255(sum_value):
    if sum_value < 0:
        return sum_value
    elif sum_value > 255:
        return 255 - (sum_value - 255)
    else:
        return sum_value


# Saturate between 0 and any poistive number
def saturated_between_0_any_number(sum_value):
    if sum_value < 0:
        return sum_value * -1
    else:
        return sum_value


# Saturate with no linmit, in fact just the number as it is
def saturated_no_limit(sum_value):
    return sum_value


# comparing two image pixels, important it compares the same pixel address
def compare_two_images(src_img_pointer, dest_image_pointer, row, col):
    difference = int(src_img_pointer[row][col]) - int(dest_image_pointer[row][col])
    if difference < 0 or difference > 255:
        print(row, col, 'difference:', difference, 'src_pixel:', src_img_pointer[row][col], 'dest_pixel:',
              dest_image_pointer[row][col], 'Saturated:', saturated_between_0_255_invert(difference))
    return saturated_between_0_255_invert((difference))


def compare_two_radial_lines(norm1_img_pointer, norm2_image_pointer, row, col):
    difference = int(norm1_img_pointer[row][col]) - int(norm2_image_pointer[row][col])

    return difference


# comparing two image pixels, important it compares the different pixel address
def compare_two_images_different_pixel(src_img_pointer, dest_image_pointer, src_row, src_col, dest_row, dest_col):
    difference = int(src_img_pointer[src_row][src_col] - dest_image_pointer[dest_row][dest_col])
    if difference < 0 or difference > 255:
        print(src_row, src_col, difference)
    return difference


# rescaling of a pixel based on min max value
def rescale_to_min_max(img_pointer, row, col, min, max):
    current_intensity = img_pointer[row, col]
    result = (current_intensity - min) * 255 / 113.265768548
    # 80.6625   #113.265768548 # this skips some values after rescaling
    # print(result)
    return result


def rescale_non_inversion(img_pointer, row, col, min, max):
    current_intensity = img_pointer[row, col]
    result = ((current_intensity - (-109.49583333333334)) / (109.19583333333331 - (-109.49583333333334)) * 255)

    # (-113.268642766)) / (112.507023317 - (-113.268642766)) * 255)

    # (-104.625)) / (103.625 - (-104.625)) * 255)

    # (-98.9999754382)) / (92.2164940402 - (-98.9999754382)) * 255)
    # (-92.1708333333)) / (83.875- (-92.1708333333)) * 255)

    # (-90.406436527)) / (88.443615997 - (-90.406436527)) * 255)

    return result


# returns the first ring of an image  with specific pixel
def first_ring(img_pointer, row, col, first_parameter, second_parameter):
    val_first_right = int(img_pointer[row, col + 1])
    val_first_left = int(img_pointer[row, col - 1])
    val_first_up = int(img_pointer[row - 1, col])
    val_first_down = int(img_pointer[row + 1, col])
    val_diag_one = int(img_pointer[row - 1, col - 1])
    val_diag_two = int(img_pointer[row - 1, col + 1])
    val_diag_three = int(img_pointer[row + 1, col + 1])
    val_diag_four = int(img_pointer[row + 1, col - 1])

    first_ring_value = val_first_right + val_first_left + val_first_up + val_first_down + val_diag_one + val_diag_two + val_diag_three + val_diag_four
    return first_ring_value / first_parameter


# returns the second ring of an image with specific pixel
def second_ring(img_pointer, row, col, first_parameter, second_parameter, third_parameter):
    val_second_right = int(img_pointer[row, col + 2])
    val_second_left = int(img_pointer[row, col - 2])
    val_second_up = int(img_pointer[row - 2, col])
    val_second_down = int(img_pointer[row + 2, col])
    val_second_diag_one = int(img_pointer[row - 2, col - 2])
    val_second_diag_two = int(img_pointer[row - 2, col + 2])
    val_second_diag_three = int(img_pointer[row + 2, col + 2])
    val_second_diag_four = int(img_pointer[row + 2, col - 2])
    second_one = int(img_pointer[row - 2, col - 1])
    second_two = int(img_pointer[row - 2, col + 1])
    second_three = int(img_pointer[row - 1, col + 2])
    second_four = int(img_pointer[row + 1, col + 2])
    second_five = int(img_pointer[row + 2, col + 1])
    second_six = int(img_pointer[row + 2, col - 1])
    second_seven = int(img_pointer[row + 1, col - 2])
    second_eight = int(img_pointer[row - 1, col - 2])

    second_ring_value = val_second_right + val_second_left + val_second_up + val_second_down + val_second_diag_one + val_second_diag_two + val_second_diag_three + val_second_diag_four + second_one + second_two + second_three + second_four + second_five + second_six + second_seven + second_eight
    return second_ring_value / first_parameter


def third_ring(img_pointer, row, col, first_parameter, second_parameter, third_parameter):
    val_third_right = int(img_pointer[row, col + 3])
    val_third_left = int(img_pointer[row, col - 3])
    val_third_up = int(img_pointer[row - 3, col])
    val_third_down = int(img_pointer[row + 3, col])
    val_third_diag_one = int(img_pointer[row - 3, col - 3])
    val_third_diag_two = int(img_pointer[row - 3, col + 3])
    val_third_diag_three = int(img_pointer[row + 3, col + 3])
    val_third_diag_four = int(img_pointer[row + 3, col - 3])
    third_one = int(img_pointer[row - 3, col - 2])
    third_two = int(img_pointer[row - 3, col - 1])
    third_three = int(img_pointer[row + 3, col + 1])
    third_four = int(img_pointer[row - 3, col + 2])
    third_five = int(img_pointer[row - 2, col + 3])
    third_six = int(img_pointer[row - 1, col + 3])
    third_seven = int(img_pointer[row + 1, col + 3])
    third_eight = int(img_pointer[row + 2, col + 3])
    third_nine = int(img_pointer[row + 3, col + 2])
    third_ten = int(img_pointer[row + 3, col + 1])
    third_eleven = int(img_pointer[row + 3, col - 1])
    third_twelve = int(img_pointer[row + 3, col - 2])
    third_thirteen = int(img_pointer[row + 2, col - 3])
    third_forteen = int(img_pointer[row + 1, col - 3])
    third_fifteen = int(img_pointer[row - 1, col - 3])
    third_sixteen = int(img_pointer[row - 2, col - 3])

    third_ring_value = int(
        (((int(val_third_right)) + int(val_third_left) + int(val_third_up) + int(val_third_down)) / first_parameter) \
        + ((int(val_third_diag_one) + int(val_third_diag_two) + int(val_third_diag_three) + int(
            val_third_diag_four)) / second_parameter) \
        + ((int(third_one) + int(third_two) + int(third_three) + int(third_four) + int(third_five) + int(
            third_six) + int(third_seven) + int(third_eight) + int(third_nine) + int(third_ten) + int(
            third_eleven) + int(third_twelve) + int(third_thirteen)
            + int(third_forteen) + int(third_fifteen) + int(third_sixteen)) / third_parameter))
    return third_ring_value


def fourth_ring(img_pointer, row, col, first_parameter, second_parameter, third_parameter):
    val_third_right = int(img_pointer[row, col + 4])
    val_third_left = int(img_pointer[row, col - 3])
    val_third_up = int(img_pointer[row - 3, col + 1])
    val_third_down = int(img_pointer[row + 2, col])
    val_third_diag_one = int(img_pointer[row - 2, col - 2])
    val_third_diag_two = int(img_pointer[row - 2, col + 2])
    val_third_diag_three = int(img_pointer[row + 2, col + 2])
    val_third_diag_four = int(img_pointer[row + 2, col - 2])
    third_one = int(img_pointer[row - 2, col - 3])
    third_two = int(img_pointer[row - 1, col - 3])
    third_three = int(img_pointer[row + 1, col - 3])
    third_four = int(img_pointer[row + 2, col - 3])
    third_five = int(img_pointer[row + 3, col - 2])
    third_six = int(img_pointer[row + 3, col - 1])
    third_seven = int(img_pointer[row + 3, col + 1])
    third_eight = int(img_pointer[row + 3, col + 2])
    third_nine = int(img_pointer[row + 2, col + 3])
    third_ten = int(img_pointer[row + 1, col + 3])
    third_eleven = int(img_pointer[row - 1, col + 3])
    third_twelve = int(img_pointer[row - 2, col + 3])
    third_thirteen = int(img_pointer[row - 3, col + 2])
    third_forteen = int(img_pointer[row - 3, col + 1])
    third_fifteen = int(img_pointer[row - 3, col - 1])
    third_sixteen = int(img_pointer[row - 3, col - 2])

    third_ring_value = int(
        (((int(val_third_right)) + int(val_third_left) + int(val_third_up) + int(val_third_down)) / first_parameter) \
        + ((int(val_third_diag_one) + int(val_third_diag_two) + int(val_third_diag_three) + int(
            val_third_diag_four)) / second_parameter) \
        + ((int(third_one) + int(third_two) + int(third_three) + int(third_four) + int(third_five) + int(
            third_six) + int(third_seven) + int(third_eight) + int(third_nine) + int(third_ten) + int(
            third_eleven) + int(third_twelve) + int(third_thirteen)
            + int(third_forteen) + int(third_fifteen) + int(third_sixteen)) / third_parameter))
    return third_ring_value


# get max intensity value in an image
def get_max_intensity(image_pointer):
    return np.amax(image_pointer)


# get min intensity value in an image
def get_min_intensity(image_pointer):
    return np.amin(image_pointer)


# get min intensity value in an image
def get_img_intensity(image_pointer, row, col):
    return int(image_pointer[row, col])


def pixel_neighbours(img, row_number, col_number, ring_number):
    # if ring_number is not 9:
    #    print(ring_number, 'do whatever you want to do')
    rows, cols = img.shape
    minimum_row = row_number - ring_number if row_number - ring_number >= 0 else 0
    # above statement is same as below, we are only using shorthand version of if/else
    # if row_number - ring_number >= 0:
    #   minimum_row = row_number - ring_number
    # else
    #   minimum_row = 0
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


def get_1st_ring(gray_img, row_number, col_number, parameter_1):
    # current_pixel_intensity = gray_img[row_number][col_number]
    # ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    # result = int(current_pixel_intensity - ring_intensity_sum_1)
    # result = saturated_between_0_255_invert(result)
    return 0


def get_2nd_ring(gray_img, row_number, col_number, parameter_1, parameter_2):
    # current_pixel_intensity = gray_img[row_number][col_number]
    # ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    # ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    # result = int(current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2)
    # result = saturated_between_0_255_invert(result)
    return 0


def get_3rd_ring(gray_img, row_number, col_number, parameter_1, parameter_2, parameter_3):
    # current_pixel_intensity = gray_img[row_number][col_number]
    # ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    # ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    # ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    # result = int(current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3)
    # result = saturated_between_0_255_invert(result)
    return 0


def get_4th_ring(gray_img, row_number, col_number, parameter_1, parameter_2, parameter_3, parameter_4, is_needed):
    if is_needed == 0:
        return 0
    if row_number == 4 and col_number == 4:
        print()
    current_pixel_intensity = gray_img[row_number][col_number]
    ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    ring_intensity_sum_4 = pixel_neighbours(gray_img, row_number, col_number, 4) / parameter_4
    result = int(
        current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3 - ring_intensity_sum_4)
    result = saturated_between_0_255_invert(result)
    return result


def get_5th_ring(gray_img, row_number, col_number, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5,
                 is_needed):
    if is_needed == 0:
        return 0
    current_pixel_intensity = gray_img[row_number][col_number]
    ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    ring_intensity_sum_4 = pixel_neighbours(gray_img, row_number, col_number, 4) / parameter_4
    ring_intensity_sum_5 = pixel_neighbours(gray_img, row_number, col_number, 5) / parameter_5
    result = int(
        current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3 - ring_intensity_sum_4 - ring_intensity_sum_5)
    result = saturated_between_0_255_invert(result)
    return result


def get_6th_ring(gray_img, row_number, col_number, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5,
                 parameter_6, is_needed):
    if is_needed == 0:
        return 0
    current_pixel_intensity = gray_img[row_number][col_number]
    ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    ring_intensity_sum_4 = pixel_neighbours(gray_img, row_number, col_number, 4) / parameter_4
    ring_intensity_sum_5 = pixel_neighbours(gray_img, row_number, col_number, 5) / parameter_5
    ring_intensity_sum_6 = pixel_neighbours(gray_img, row_number, col_number, 6) / parameter_6
    result = int(
        current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3 - ring_intensity_sum_4 - ring_intensity_sum_5 - ring_intensity_sum_6)
    result = saturated_between_0_255_invert(result)
    return result


def get_7th_ring(gray_img, row_number, col_number, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5,
                 parameter_6, parameter_7):
    # current_pixel_intensity = gray_img[row_number][col_number]
    # ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    # ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    # ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    # ring_intensity_sum_4 = pixel_neighbours(gray_img, row_number, col_number, 4) / parameter_4
    # ring_intensity_sum_5 = pixel_neighbours(gray_img, row_number, col_number, 5) / parameter_5
    # ring_intensity_sum_6 = pixel_neighbours(gray_img, row_number, col_number, 6) / parameter_6
    # ring_intensity_sum_7 = pixel_neighbours(gray_img, row_number, col_number, 7) / parameter_7
    # result = int(
    #    current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3 - ring_intensity_sum_4 - ring_intensity_sum_5 - ring_intensity_sum_6 - ring_intensity_sum_7)
    # result = saturated_between_0_255_invert(result)
    return 0


def get_8th_ring(gray_img, row_number, col_number, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5,
                 parameter_6, parameter_7, parameter_8):
    # current_pixel_intensity = gray_img[row_number][col_number]
    # ring_intensity_sum_1 = pixel_neighbours(gray_img, row_number, col_number, 1) / parameter_1
    # ring_intensity_sum_2 = pixel_neighbours(gray_img, row_number, col_number, 2) / parameter_2
    # ring_intensity_sum_3 = pixel_neighbours(gray_img, row_number, col_number, 3) / parameter_3
    # ring_intensity_sum_4 = pixel_neighbours(gray_img, row_number, col_number, 4) / parameter_4
    # ring_intensity_sum_5 = pixel_neighbours(gray_img, row_number, col_number, 5) / parameter_5
    # ring_intensity_sum_6 = pixel_neighbours(gray_img, row_number, col_number, 6) / parameter_6
    # ring_intensity_sum_7 = pixel_neighbours(gray_img, row_number, col_number, 7) / parameter_7
    # ring_intensity_sum_8 = pixel_neighbours(gray_img, row_number, col_number, 8) / parameter_8
    # result = int(
    #     current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3 - ring_intensity_sum_4 - ring_intensity_sum_5 - ring_intensity_sum_6 - ring_intensity_sum_7 - ring_intensity_sum_8)
    # result = saturated_between_0_255_invert(result)
    return 0


def get_9th_ring(gray_img, row_number, col_number, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5,
                 parameter_6, parameter_7, parameter_8, parameter_9):
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
    result = int(
        current_pixel_intensity - ring_intensity_sum_1 - ring_intensity_sum_2 - ring_intensity_sum_3 - ring_intensity_sum_4 - ring_intensity_sum_5 - ring_intensity_sum_6 - ring_intensity_sum_7 - ring_intensity_sum_8 - ring_intensity_sum_9)
    result = saturated_between_0_255_invert(result)
    return result


def pixel_calculator_4(src, parameter_1, parameter_2, parameter_3, parameter_4):
    src_image = np.copy(src)
    height, width = src_image.shape  # defines grayscale(2D) image height and width
    print('The original image is of dimensions:', src_image.shape, ' data_type:', src_image.dtype, ' min_value:',
          np.amin(src_image), ' max_value:', np.amax(src_image))
    # print(src_image)

    new_img = np.zeros(src_image.shape, src_image.dtype)  # calculated matrix made of equivalent dimension and data type

    total_rows, total_cols = src.shape  # defines grayscale(2D) image height and width
    calc = 0
    # loop through whole image except 1st row,col,last row and col. they are set to zero
    for row_number in range(1, total_rows - 1):
        for col_number in range(1, total_cols - 1):
            # print(row_number, col_number)

            if row_number == 1 or col_number == 1 or row_number == total_rows - 2 or col_number == total_cols - 2:
                new_intensity = get_1st_ring(src_image, row_number, col_number, parameter_1)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 2 or col_number == 2 or row_number == total_rows - 3 or col_number == total_cols - 3:
                new_intensity = get_2nd_ring(src_image, row_number, col_number, parameter_1, parameter_2)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 3 or col_number == 3 or row_number == total_rows - 4 or col_number == total_cols - 4:
                new_intensity = get_3rd_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 4 or col_number == 4 or row_number == total_rows - 5 or col_number == total_cols - 5:
                new_intensity = get_4th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, 1)
                new_img[row_number][col_number] = new_intensity
            else:
                new_intensity = get_4th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, 1)
                new_img[row_number][col_number] = new_intensity

    return new_img


def pixel_calculator_5(src, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5):
    src_image = np.copy(src)
    height, width = src_image.shape  # defines grayscale(2D) image height and width
    print('The original image is of dimensions:', src_image.shape, ' data_type:', src_image.dtype, ' min_value:',
          np.amin(src_image), ' max_value:', np.amax(src_image))
    # print(src_image)

    new_img = np.zeros(src_image.shape, src_image.dtype)  # calculated matrix made of equivalent dimension and data type

    total_rows, total_cols = src.shape  # defines grayscale(2D) image height and width
    calc = 0
    # loop through whole image except 1st row,col,last row and col. they are set to zero
    for row_number in range(1, total_rows - 1):
        for col_number in range(1, total_cols - 1):
            # print(row_number, col_number)

            if row_number == 1 or col_number == 1 or row_number == total_rows - 2 or col_number == total_cols - 2:
                new_intensity = get_1st_ring(src_image, row_number, col_number, parameter_1)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 2 or col_number == 2 or row_number == total_rows - 3 or col_number == total_cols - 3:
                new_intensity = get_2nd_ring(src_image, row_number, col_number, parameter_1, parameter_2)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 3 or col_number == 3 or row_number == total_rows - 4 or col_number == total_cols - 4:
                new_intensity = get_3rd_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 4 or col_number == 4 or row_number == total_rows - 5 or col_number == total_cols - 5:
                new_intensity = get_4th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, 0)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 5 or col_number == 5 or row_number == total_rows - 6 or col_number == total_cols - 6:
                new_intensity = get_5th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, 1)
                new_img[row_number][col_number] = new_intensity
            else:
                new_intensity = get_5th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, 1)
                new_img[row_number][col_number] = new_intensity

    return new_img


def pixel_calculator_6(src, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6):
    src_image = np.copy(src)
    height, width = src_image.shape  # defines grayscale(2D) image height and width
    print('The original image is of dimensions:', src_image.shape, ' data_type:', src_image.dtype, ' min_value:',
          np.amin(src_image), ' max_value:', np.amax(src_image))
    # print(src_image)

    new_img = np.zeros(src_image.shape, src_image.dtype)  # calculated matrix made of equivalent dimension and data type

    total_rows, total_cols = src.shape  # defines grayscale(2D) image height and width
    calc = 0
    # loop through whole image except 1st row,col,last row and col. they are set to zero
    for row_number in range(1, total_rows - 1):
        for col_number in range(1, total_cols - 1):
            # print(row_number, col_number)

            if row_number == 1 or col_number == 1 or row_number == total_rows - 2 or col_number == total_cols - 2:
                new_intensity = get_1st_ring(src_image, row_number, col_number, parameter_1)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 2 or col_number == 2 or row_number == total_rows - 3 or col_number == total_cols - 3:
                new_intensity = get_2nd_ring(src_image, row_number, col_number, parameter_1, parameter_2)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 3 or col_number == 3 or row_number == total_rows - 4 or col_number == total_cols - 4:
                new_intensity = get_3rd_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 4 or col_number == 4 or row_number == total_rows - 5 or col_number == total_cols - 5:
                new_intensity = get_4th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, 0)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 5 or col_number == 5 or row_number == total_rows - 6 or col_number == total_cols - 6:
                new_intensity = get_5th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, 0)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 6 or col_number == 6 or row_number == total_rows - 7 or col_number == total_cols - 7:
                new_intensity = get_6th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, 1)
                new_img[row_number][col_number] = new_intensity
            else:
                new_intensity = get_6th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, 1)
                new_img[row_number][col_number] = new_intensity

    return new_img


def pixel_calculator_7(src, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7):
    src_image = np.copy(src)
    height, width = src_image.shape  # defines grayscale(2D) image height and width
    print('The original image is of dimensions:', src_image.shape, ' data_type:', src_image.dtype, ' min_value:',
          np.amin(src_image), ' max_value:', np.amax(src_image))
    # print(src_image)

    new_img = np.zeros(src_image.shape, src_image.dtype)  # calculated matrix made of equivalent dimension and data type

    total_rows, total_cols = src.shape  # defines grayscale(2D) image height and width
    calc = 0
    # loop through whole image except 1st row,col,last row and col. they are set to zero
    for row_number in range(1, total_rows - 1):
        for col_number in range(1, total_cols - 1):
            # print(row_number, col_number)

            if row_number == 1 or col_number == 1 or row_number == total_rows - 2 or col_number == total_cols - 2:
                new_intensity = get_1st_ring(src_image, row_number, col_number, parameter_1)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 2 or col_number == 2 or row_number == total_rows - 3 or col_number == total_cols - 3:
                new_intensity = get_2nd_ring(src_image, row_number, col_number, parameter_1, parameter_2)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 3 or col_number == 3 or row_number == total_rows - 4 or col_number == total_cols - 4:
                new_intensity = get_3rd_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 4 or col_number == 4 or row_number == total_rows - 5 or col_number == total_cols - 5:
                new_intensity = get_4th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, 0)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 5 or col_number == 5 or row_number == total_rows - 6 or col_number == total_cols - 6:
                new_intensity = get_5th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, 0)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 6 or col_number == 6 or row_number == total_rows - 7 or col_number == total_cols - 7:
                new_intensity = get_6th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, 0)
            elif row_number == 7 or col_number == 7 or row_number == total_rows - 8 or col_number == total_cols - 8:
                new_intensity = get_7th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, parameter_7)
                new_img[row_number][col_number] = new_intensity
            else:
                new_intensity = get_7th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, parameter_7)
                new_img[row_number][col_number] = new_intensity

    return new_img


def pixel_calculator_8(src, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7,
                       parameter_8):
    src_image = np.copy(src)
    height, width = src_image.shape  # defines grayscale(2D) image height and width
    print('The original image is of dimensions:', src_image.shape, ' data_type:', src_image.dtype, ' min_value:',
          np.amin(src_image), ' max_value:', np.amax(src_image))
    # print(src_image)

    new_img = np.zeros(src_image.shape, src_image.dtype)  # calculated matrix made of equivalent dimension and data type

    total_rows, total_cols = src.shape  # defines grayscale(2D) image height and width
    calc = 0
    # loop through whole image except 1st row,col,last row and col. they are set to zero
    for row_number in range(1, total_rows - 1):
        for col_number in range(1, total_cols - 1):
            # print(row_number, col_number)

            if row_number == 1 or col_number == 1 or row_number == total_rows - 2 or col_number == total_cols - 2:
                new_intensity = get_1st_ring(src_image, row_number, col_number, parameter_1)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 2 or col_number == 2 or row_number == total_rows - 3 or col_number == total_cols - 3:
                new_intensity = get_2nd_ring(src_image, row_number, col_number, parameter_1, parameter_2)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 3 or col_number == 3 or row_number == total_rows - 4 or col_number == total_cols - 4:
                new_intensity = get_3rd_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 4 or col_number == 4 or row_number == total_rows - 5 or col_number == total_cols - 5:
                new_intensity = get_4th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, 0)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 5 or col_number == 5 or row_number == total_rows - 6 or col_number == total_cols - 6:
                new_intensity = get_5th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, 0)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 6 or col_number == 6 or row_number == total_rows - 7 or col_number == total_cols - 7:
                new_intensity = get_6th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, 0)
            elif row_number == 7 or col_number == 7 or row_number == total_rows - 8 or col_number == total_cols - 8:
                new_intensity = get_7th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, parameter_7)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 8 or col_number == 8 or row_number == total_rows - 9 or col_number == total_cols - 9:
                new_intensity = get_8th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, parameter_7, parameter_8)
                new_img[row_number][col_number] = new_intensity
            # elif row_number == 9 or col_number == 9 or row_number == total_rows - 10 or col_number == total_cols - 10:
            #     new_intensity = get_9th_ring(src_image, row_number, col_number)
            #     new_img[row_number][col_number] = new_intensity
            else:
                new_intensity = get_8th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, parameter_7, parameter_8)
                new_img[row_number][col_number] = new_intensity

    return new_img


def pixel_calculator_9(src, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7,
                       parameter_8, parameter_9):
    src_image = np.copy(src)
    height, width = src_image.shape  # defines grayscale(2D) image height and width
    print('The original image is of dimensions:', src_image.shape, ' data_type:', src_image.dtype, ' min_value:',
          np.amin(src_image), ' max_value:', np.amax(src_image))
    # print(src_image)

    new_img = np.zeros(src_image.shape, src_image.dtype)  # calculated matrix made of equivalent dimension and data type

    total_rows, total_cols = src.shape  # defines grayscale(2D) image height and width
    calc = 0
    # loop through whole image except 1st row,col,last row and col. they are set to zero
    for row_number in range(1, total_rows - 1):
        for col_number in range(1, total_cols - 1):
            # print(row_number, col_number)

            if row_number == 1 or col_number == 1 or row_number == total_rows - 2 or col_number == total_cols - 2:
                new_intensity = get_1st_ring(src_image, row_number, col_number, parameter_1)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 2 or col_number == 2 or row_number == total_rows - 3 or col_number == total_cols - 3:
                new_intensity = get_2nd_ring(src_image, row_number, col_number, parameter_1, parameter_2)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 3 or col_number == 3 or row_number == total_rows - 4 or col_number == total_cols - 4:
                new_intensity = get_3rd_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 4 or col_number == 4 or row_number == total_rows - 5 or col_number == total_cols - 5:
                new_intensity = get_4th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, 0)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 5 or col_number == 5 or row_number == total_rows - 6 or col_number == total_cols - 6:
                new_intensity = get_5th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, 0)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 6 or col_number == 6 or row_number == total_rows - 7 or col_number == total_cols - 7:
                new_intensity = get_6th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, 0)
            elif row_number == 7 or col_number == 7 or row_number == total_rows - 8 or col_number == total_cols - 8:
                new_intensity = get_7th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, parameter_7)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 8 or col_number == 8 or row_number == total_rows - 9 or col_number == total_cols - 9:
                new_intensity = get_8th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, parameter_7, parameter_8)
                new_img[row_number][col_number] = new_intensity
            elif row_number == 9 or col_number == 9 or row_number == total_rows - 10 or col_number == total_cols - 10:
                new_intensity = get_9th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, parameter_7, parameter_8,
                                             parameter_9)
                new_img[row_number][col_number] = new_intensity
            else:
                new_intensity = get_9th_ring(src_image, row_number, col_number, parameter_1, parameter_2, parameter_3,
                                             parameter_4, parameter_5, parameter_6, parameter_7, parameter_8,
                                             parameter_9)
                new_img[row_number][col_number] = new_intensity

    return new_img
