import os
import sys

import cv2 as cv
import numpy as np

from modules import coreimagemodule

root = os.getcwd()
print('root path: ', root)


def make_circle(img, circle_x_position, circle_y_position):
    print(img.shape)
    src_height, src_width = img.shape

    I = []
    for c in range(1,
                   src_height - circle_y_position):
        # integrate
        val = my_func(img, circle_x_position, circle_y_position, c)
        result = val * c

        I.append(result)

    I = np.array(I)
    print(I)


def my_func(img, circle_x, circle_y, c):
    print(circle_x, circle_y + c)
    row = circle_x
    col = circle_y + c
    current_intensity = int(img[row, col])

    return (current_intensity * 2 * 3.14 * c) / (3.14 * (c ** 2))


def main():
    root_path = root + '\\Images\\New_image\\ring_2\\Compare\res8_res_-2_+16by3.png'
    img = cv.imread(root_path)
    if img is None:
        print('Failed to load image:')
        sys.exit(1)
    src = coreimagemodule.color_to_grayscale(img)

    make_circle(src, 50, 50)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0


# Now we call the main function
main()
