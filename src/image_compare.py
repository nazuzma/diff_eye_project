import os
import sys

import cv2 as cv
import numpy as np

from modules import coreimagemodule

root = os.getcwd()


def rescale(src):
    src_image = np.copy(src)
    height, width = src_image.shape  # defines grayscale(2D) image height and width
    print(src_image)
    max = np.amax(src_image)
    min = np.amin(src_image)
    print('min is:', min, ' max is:', max)

    matrix = np.zeros(src_image.shape, src_image.dtype)  # calculated matrix made of equivalent dimension and data type
    calc = 0

    # loop which goes through all required pixels
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            matrix[j, i] = coreimagemodule.rescale_to_min_max(src_image, j, i, min, max)


def main():
    src_image_path = root + 'input_images//img_1.jpg'  # try src.png
    dest_image_path = root + 'input_images//img_2.jpg'  # try dest.png
    src = cv.imread(src_image_path)
    dest = cv.imread(dest_image_path)
    if src is None or dest is None:
        print('Failed to load image:')
        sys.exit(1)
    src_image = coreimagemodule.color_to_grayscale(src)
    dest_image = coreimagemodule.color_to_grayscale(dest)
    src_height, src_width = src_image.shape
    dest_height, dest_width = dest_image.shape
    print("src and dest shape", src_image.shape, dest_image.shape)
    if src_height != dest_height or src_width != dest_width:
        print('Sorry images does not have same size, please try again')
        sys.exit(1)

    matrix = np.zeros(src_image.shape, src_image.dtype)

    calc = 0

    # loop which goes through all required pixels
    for j in range(1, src_height - 1):
        for i in range(1, src_width - 1):
            matrix[j, i] = coreimagemodule.compare_two_images(src_image, dest_image, j,
                                                              i)  # using compare function on each pixel
            if matrix[j, i] == 255:
                print('img[{0},{1}]:{2}'.format(j, i, matrix[j, i]))
    rescale(matrix)
    # since we don't have value greater than 255 hence Rescaled value is same as compare_two_images result

    print(matrix)
    print('min of SRC is:', np.amin(src_image), ' max of SRC is :', np.amax(src_image))
    print('min of DEST is:', np.amin(dest_image), ' max of DEST is:', np.amax(dest_image))
    print('New min is:', np.amin(matrix), ' New max is:', np.amax(matrix))
    cv.namedWindow("Calculated", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Rescaled", cv.WINDOW_AUTOSIZE)
    cv.imshow("Calculated", matrix)  # displays calculated image
    cv.imshow("Rescaled", matrix)  # displays Rescaled image

    cv.imwrite('output_images//compare_1st_-9_-144.png', matrix)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0


# Now we call the main function
main()
print('program end')
sys.exit(1)
