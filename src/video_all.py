import os
import sys

import cv2
import numpy as np

root = os.getcwd()
root_path = root + '\\video\\Traffic.avi'


def get_derivative_and_modify_frame(previous_frame_gray, current_frame, current_frame_number):
    height, width = current_frame.shape  # defines grayscale(2D) image height and width
    new_current_frame_gray = np.copy(current_frame)
    ##print(new_current_frame_gray)

    calc = 0

    # loop which goes through all required pixels
    for j in range(1, height - 1):

        for i in range(1, width - 1):
            calc += 1
            current_y = int(current_frame[j, i])
            current_x = current_frame_number
            prev_x = current_frame_number - 1
            prev_y = int(previous_frame_gray[j, i])
            res = int((current_y - prev_y) / (current_x - prev_x))
            if res > 50:
                res = 255
            elif res < -50:
                res = 255
            else:
                res = 0
            new_current_frame_gray[j, i] = res

    return new_current_frame_gray


def read_video_and_get_all_derivative_and_save():
    previous_frame_gray = None
    # current_frame_gray_new = None
    """
    This function saves video after receiving derivative array which need to be put as new intensity of that pixel
    """
    # Create a VideoCapture object
    video = cv2.VideoCapture(root_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Unable to read video")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    timestamps_x_axis = []
    duration = (total_frames / fps) * 1000  # in ms

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    print("total_frames:{0}, width:{1}, Height:{2}, FPS:{3}, duration:{4}".format(total_frames, width, height, fps,
                                                                                  duration))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(root + '\\video\\test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          (width, height), 0)
    current_frame_number = 1
    """
     before reading each frame from video we initialize it, so that we can increase frame number after reading each next Frame
    """

    while True:  # if statement id true, execute these set of statements
        is_frame_exists, frame = video.read()
        current_frame_time = int(video.get(cv2.CAP_PROP_POS_MSEC))  # extracting time span

        if is_frame_exists:
            current_frame_gray = cv2.cvtColor(frame,
                                              cv2.COLOR_BGR2GRAY)  # changing original frames to grayscale as input video is color
            # print(current_frame_gray)

            if previous_frame_gray is None:
                previous_frame_gray = current_frame_gray  # saves previous frame

                # print(previous_frame_gray)

            # Write the frame into the file 'output.avi'
            # we save each frame in output video using out.write(frame)
            # modify_frame() function returns modified frame by manipulating intensities of specific pixels
            cv2.imshow('frame', current_frame_gray)
            # print(current_frame_number)

            modified_frame = get_derivative_and_modify_frame(previous_frame_gray, current_frame_gray,
                                                             current_frame_number)

            out.write(modified_frame)  # save output video

            # Display the resulting frame
            cv2.imshow('modified_frame', modified_frame)

            # print(previous_frame_gray)
            # print(current_frame_gray)

            previous_frame_gray = current_frame_gray

            current_frame_number = current_frame_number + 1

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

        # When everything done, release the video capture and video write objects
    video.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def main():
    read_video_and_get_all_derivative_and_save()


main()
print('program end')
sys.exit(1)
