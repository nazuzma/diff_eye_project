import os
import sys

import cv2
import matplotlib.pyplot as plt

root = os.getcwd()
root_path = root + '\\input_videos\\Traffic.avi'

video = cv2.VideoCapture(root_path)

if video is None:  # ensure that video has uploaded
    print('Failed to load video:')
    sys.exit(1)


def get_derivative_with_previous(x_array, y_array):
    count = 1
    diff = []
    while count < len(y_array):
        current_y = int(y_array[count])
        current_x = int(x_array[count])
        prev_x = int(x_array[count - 1])
        prev_y = int(y_array[count - 1])
        res = int((current_y - prev_y) / (current_x - prev_x))
        diff.append(res)
        count = count + 1
    return diff


def get_derivative_with_next(x_array, y_array):
    count = 0
    diff = []
    while count < len(y_array) - 1:
        current_y = int(y_array[count])
        current_x = int(x_array[count])
        next_x = int(x_array[count + 1])
        next_y = int(y_array[count + 1])
        res = int((next_y - current_y) / (next_x - current_x))
        diff.append(res)
        count = count + 1
    return diff


def get_derivative_with_center(x_array, y_array):
    count = 0
    diff = []
    print(len(y_array))
    while count < len(y_array) - 1:
        current_y = int(y_array[count])
        current_x = int(x_array[count])
        next_x = int(x_array[count + 1])
        next_y = int(y_array[count + 1])
        prev_x = int(x_array[count - 1])
        prev_y = int(y_array[count - 1])
        res = int((next_y - prev_y) / (next_x - prev_x))
        if res > 20:
            res = 255
        elif res < -20:
            res = 255
        else:
            res = 0
        diff.append(res)
        count = count + 1

    return diff


def get_derivative_with_center1(x_array, y_array):  # function without condition
    count = 0
    diff = []
    print(len(y_array))
    while count < len(y_array) - 1:
        current_y = int(y_array[count])
        current_x = int(x_array[count])
        next_x = int(x_array[count + 1])
        next_y = int(y_array[count + 1])
        prev_x = int(x_array[count - 1])
        prev_y = int(y_array[count - 1])
        res = int((next_y - prev_y) / (next_x - prev_x))
        diff.append(res)
        count = count + 1

    return diff


def save_list_to_text_file(file_name, string_to_write):
    separate_lines_text = '\n'.join(string_to_write)
    text_file = open("{0}".format(file_name), "w")
    text_file.write(separate_lines_text)
    text_file.close()
    print("Saving file: '{0}' Successful".format(file_name))


def main():
    frame_number = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    timestamps_x_axis = []
    duration = (total_frames / fps) * 1000  # in ms
    list_color_text = []
    list_gray_text = []
    list_bw_text = []
    frame_axis = []
    intensity_y_axis = []  # creating list for pixel values
    pixel_row_number = 260
    pixel_col_number = 310
    print("total_frames:{0}, width:{1}, Height:{2}, FPS:{3}, duration:{4}".format(total_frames, width, height, fps,
                                                                                  duration))

    # loop over the frames of the video
    while video.isOpened():
        # grab the current frame
        is_frame_exists, current_frame = video.read()

        # check to see if we have reached the end of the video

        if is_frame_exists:
            frame_axis.append(frame_number)
            current_frame_time = int(video.get(cv2.CAP_PROP_POS_MSEC))  # extracting time span
            timestamps_x_axis.append(current_frame_time)  # adding items in time matrix
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)  # changing original frames to grayscale
            thresh, blackAndWhiteImage = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # if you want grayscale then replace current_frame with gray similar with blackAndWhite
            cv2.imshow('frame ' + str(frame_number), current_frame)
            # cv2.waitKey(10)

            intensity_y_axis.append(gray[pixel_row_number][pixel_col_number])

            list_color_text.append(
                "Timestamp(ms)#:{0}  , Pixel Address:[{1}][{2}] ,  {3}".format(current_frame_time, pixel_row_number,
                                                                               pixel_col_number,
                                                                               current_frame[pixel_row_number][
                                                                                   pixel_col_number]))
            list_gray_text.append(
                "Timestamp(ms)#:{0}  , Pixel Address:[{1}][{2}] ,  {3}".format(current_frame_time, pixel_row_number,
                                                                               pixel_col_number,
                                                                               gray[pixel_row_number][
                                                                                   pixel_col_number]))
            list_bw_text.append(
                "Timestamp(ms)#:{0}  , Pixel Address:[{1}][{2}] ,  {3}".format(current_frame_time, pixel_row_number,
                                                                               pixel_col_number,
                                                                               blackAndWhiteImage[pixel_row_number][
                                                                                   pixel_col_number]))
            # print("Frame#:{0}  , Pixel Address:[15][16] ,  {1}".format(count, current_frame[15][16]))
        else:
            break

        frame_number = frame_number + 1  # increment the total number of frames read

    save_list_to_text_file("Color Output.txt", list_color_text)
    save_list_to_text_file("Grayscale Output.txt", list_gray_text)
    save_list_to_text_file("BW Output.txt", list_bw_text)

    derivative_with_center = get_derivative_with_center(frame_axis, intensity_y_axis)
    derivative_with_center1 = get_derivative_with_center1(frame_axis, intensity_y_axis)
    derivative_with_prev = get_derivative_with_previous(frame_axis, intensity_y_axis)
    derivative_with_next = get_derivative_with_next(frame_axis, intensity_y_axis)
    print("prev Y:", derivative_with_prev)
    print("next Y:", derivative_with_next)
    print("center Y:", derivative_with_center)
    print("Y:prev,next,center:  {0} {1} {2}".format(len(derivative_with_prev), len(derivative_with_next),
                                                    len(derivative_with_center)))

    # print(timestamps_x_axis)
    # print(len(timestamps_x_axis))
    # print(diff
    new_x_ax = timestamps_x_axis.copy()
    new_x_ax.pop(0)  # removing first point to equate both x and y arrays

    list_derivative = []  # making text file for derivatives
    for i in range(1, len(new_x_ax) + 1):
        list_derivative.append("X:{0} Y:{1}".format(new_x_ax[i - 1], derivative_with_prev[i - 1]))
    save_list_to_text_file("Derivative.txt", list_derivative)

    x = []
    for i in range(1, len(derivative_with_center) + 1):
        x.append(new_x_ax[i - 1])

    print("X:prev,next,center:  {0} {1} {2}".format(len(new_x_ax), len(new_x_ax),
                                                    len(x)))
    print("prev x:", new_x_ax)
    print("next x:", new_x_ax)
    print("center x:", x)

    plt.plot(timestamps_x_axis, intensity_y_axis, label='Intensity with time')
    plt.plot(x, derivative_with_center, label='Diff with cutoff')
    plt.plot(x, derivative_with_center1, label='Diff without cutoff')
    plt.legend()

    # plt.plot(new_x_ax, derivative_with_prev, label='Diff prev')
    # plt.plot(new_x_ax, derivative_with_next, label='Diff next')
    ax = plt.axes()
    ax.yaxis.grid()
    plt.xlabel('Timestamp t(ms)')
    plt.ylabel('Grayscale Intensity I')
    plt.savefig('Traff.png')

    plt.show()

    video.release()
    cv2.destroyAllWindows()  # destroy all the opened windows


main()
print('program end')
sys.exit(1)
