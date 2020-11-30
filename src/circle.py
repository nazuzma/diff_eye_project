import math
import os
import sys
from statistics import mean

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from modules import coreimagemodule

root = os.getcwd()


def get_max_radius(w, h, cx, cy):
    if w == h:
        return h - cy - 1
    elif w > h:
        return h - cy - 1
    else:
        return w - cx - 1


def get_pixels_intensity(mask, img, j, size, row_or_col):
    intensity = []
    print(img.shape, size, j, row_or_col)
    for i in range(1, size - 1):
        if row_or_col == 'r':
            if mask[j][i]:
                # print(j, i, row_or_col)
                intensity.append(img[j][i])
        else:
            if mask[i][j]:
                # print(i, j, row_or_col)
                intensity.append(img[i][j])

    return intensity


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# this algorithm is getting circle of any radius by estimating star diamond like edges

def create_circular_mask(test_image, center=None, radius=None):
    masked_img = test_image.copy()
    h, w = test_image.shape[:2]
    print(h, w)
    radius = 2
    if center is None:  # use the middle of the image
        center = [int(w / 2), int(h / 2)]
    cx, cy = center

    max_radius = get_max_radius(w, h, cx, cy)  # the maximum allowed radius is (width + height)/2 so nit is average
    total_mean = []
    for i in range(2, max_radius - 1):
        print(i)
        radius = i
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = dist_from_center <= radius
        # print("w, h,center,radius ", w, h, center, radius)
        d = mask == True  # First know which array values are masked
        rows, columns = np.where(d)  # Get the positions of row and column of masked values
        rows.sort()  # sort the row values
        columns.sort()  # sort the column values
        print('Row values :', (rows[0], rows[-1]))  # print the first and last rows
        print('Column values :', (columns[0], columns[-1]))  # print the first and last columns
        first = get_pixels_intensity(mask, test_image, rows[0], w, 'r')
        first_mean = mean(first)
        second = get_pixels_intensity(mask, test_image, rows[-1], w, 'r')
        second_mean = mean(second)
        third = get_pixels_intensity(mask, test_image, columns[0], h, 'c')
        third_mean = mean(third)
        fourth = get_pixels_intensity(mask, test_image, columns[-1], h, 'c')
        print(fourth)
        fourth_mean = mean(fourth)
        total_mean.append(mean([first_mean, second_mean, third_mean, fourth_mean]))
    return total_mean


def make_circle(test_image):
    img = test_image.copy()
    h, w = img.shape

    x = np.arange(0, w)
    y = np.arange(0, h)
    arr = np.copy(img)

    cx = int(w / 2.)
    cy = int(h / 2.)
    max_radius = get_max_radius(w, h, cx, cy)
    print(w, h, cx, cy, max_radius)
    total_mean = []
    for r in range(2, max_radius):
        mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2  # equation of circle
        d = mask == True  # First know which array values are masked
        rows, columns = np.where(d)  # Get the positions of row and column of masked values

        rows.sort()  # sort the row values
        columns.sort()  # sort the column values
        first = get_pixels_intensity(mask, test_image, rows[0], w, 'r')
        first_mean = mean(first)
        second = get_pixels_intensity(mask, test_image, rows[-1], w, 'r')
        second_mean = mean(second)
        third = get_pixels_intensity(mask, test_image, columns[0], h, 'c')
        third_mean = mean(third)
        fourth = get_pixels_intensity(mask, test_image, columns[-1], h, 'c')
        fourth_mean = mean(fourth)
        total = mean([first_mean, second_mean, third_mean, fourth_mean])
        total_mean.append(total)
    return total_mean


def main():
    test_image_path = root + "input_images\\img_1.jpg"  # try src.png
    test = cv.imread(test_image_path)
    if test is None:
        print('Failed to load image:')
        sys.exit(1)
    test_image = coreimagemodule.color_to_grayscale(test)
    print(
        "#2 algorithms which takes square like pixels.....")
    calculated_means1 = make_circle(test_image)
    x1 = [*range(2, len(calculated_means1) + 2)]
    norm = (calculated_means1 - np.amin(calculated_means1)) / (
            np.amax(calculated_means1) - (np.amin(calculated_means1)))
    plt.plot(x1, norm)
    plt.xlabel('Distance from center (pixels)')
    plt.ylabel('Intensity (counts per pixel)')
    plt.title('Radial profile of Rescaled Image (1st neighbour)')
    plt.grid(True)
    plt.savefig('output_images\\res_radial.png')
    plt.show()

    print(calculated_means1)
    print(len(calculated_means1))

    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0


# Now we call the main function
main()
print('program end')
sys.exit(1)
