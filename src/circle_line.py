import os
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from modules import coreimagemodule

root = os.getcwd()


def plot_line(img, row, col, limit):
    print(row, col, limit, img.shape[:2])
    count = 1
    c = 1
    x = []
    y = []
    while count <= limit:
        y.append(img[row][col + count])
        x.append(c)
        count = count + 1
        c = c + 1
    print(x)
    print(y)
    print(len(x))
    print(len(y))
    plt.plot(x, y)
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Grayscale value(counts per pixel)')
    plt.title('Line profile of Rescaled 1st neighbour Image')
    plt.grid(True)
    plt.show()


def get_max_radius(w, h, cx, cy):
    if w == h:
        return h - cy - 1
    elif w > h:
        return h - cy - 1
    else:
        return w - cx - 1


def make_circle(test_image):
    img = test_image.copy()
    h, w = img.shape

    x = np.arange(0, w)
    y = np.arange(0, h)
    arr = np.copy(img)

    cx = int(w / 2.)
    cy = int(h / 2.)
    max_radius = get_max_radius(w, h, cx, cy)
    line_length = int((max_radius + 50) / 4)
    plot_line(img, cx, cy, line_length)
    plot_line(img, cx, cy + line_length, line_length)
    plot_line(img, cx, cy + line_length + line_length, line_length)
    plot_line(img, cx, cy + line_length + line_length + line_length, line_length)


def main():
    test_image_path = root + "input_images\\img_1.jpg"  # try src.png
    # test_image_path = root + "//Images//Test3b.JPG"
    test = cv.imread(test_image_path)
    if test is None:
        print('Failed to load image:')
        sys.exit(1)
    test_image = coreimagemodule.color_to_grayscale(test)
    make_circle(test_image)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0


# Now we call the main function
main()
print('program end')
sys.exit(1)
