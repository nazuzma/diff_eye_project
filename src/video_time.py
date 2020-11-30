import os
import sys

import cv2
import numpy as np

root = os.getcwd()
root_path = root + '\\video\\Traffic.avi'


def get_derivative_and_modify_frame(previous_frame_gray, current_frame_gray, current_frame_number):
    height, width = current_frame_gray.shape  # defines grayscale(2D) image height and width
    calc = 0

    current_frame_gray_new = np.copy(current_frame_gray)
    for j in range(0, height):
        for i in range(0, width):
            calc += 1
            current_y = int(current_frame_gray[j, i])
            # print(current_y)
            current_x = current_frame_number
            prev_x = current_frame_number - 1
            prev_y = int(previous_frame_gray[j, i])
            # print(prev_y)
            res = int((current_y - prev_y) / (current_x - prev_x))
            if res > 50:
                res = 255
            elif res < -50:
                res = 255
            else:
                res = 0
            current_frame_gray_new[j, i] = res

            # print(res)

    return current_frame_gray_new


def cutoff_value(current_frame_gray):
    height, width = current_frame_gray.shape  # defines grayscale(2D) image height and width
    calc = 0

    current_frame_gray_new = np.copy(current_frame_gray)
    for j in range(0, height):
        for i in range(0, width):
            calc += 1
            intensity = int(current_frame_gray[j, i])
            if intensity > 100:
                intensity = 255
            elif intensity < -100:
                intensity = 255
            else:
                intensity = 0
            current_frame_gray_new[j, i] = intensity

            # print(res)

    return current_frame_gray_new


def read_video_and_get_all_derivative_and_save():
    previous_frame_gray = None
    skip_frames_replacement = None
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
    duration = (total_frames / fps) * 1000  # total duration of video file in miliseconds
    current_frame_number = 0
    read_frames_time_in_ms = 50  # time in miliseconds at which u want to read video, it will be 100,200,300,400....
    new_fps = int(1000 / read_frames_time_in_ms)
    skip_frames = int(fps / new_fps)

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    print("total_frames:{0}, width:{1}, Height:{2}, FPS:{3}, duration:{4}".format(total_frames, width, height,
                                                                                  fps,
                                                                                  duration))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # int(1000/read_frames_time_in_ms) we are doing this because output video will have fps based on how many frames we read in a second

    out = cv2.VideoWriter(root + '\\video\\cuttoff100_readFrame50ms.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          fps,
                          (width, height), 0)

    """
     before reading each frame from video we initialize it, so that we can increase frame number after reading each next Frame
    """

    while True:  # if statement id true, execute these set of statements
        is_frame_exists, frame = video.read()
        current_frame_time = int(video.get(cv2.CAP_PROP_POS_MSEC))  # extracting time span

        if is_frame_exists:
            # print(current_frame_time)
            current_frame_gray = cv2.cvtColor(frame,
                                              cv2.COLOR_BGR2GRAY)  # changing original frames to grayscale as input video is color
            if previous_frame_gray is None:
                previous_frame_gray = current_frame_gray  # saves previous frame
            cv2.imshow('frame', current_frame_gray)

            if current_frame_number % skip_frames == 0:
                modified_frame = get_derivative_and_modify_frame(previous_frame_gray, current_frame_gray,
                                                                 current_frame_number)
                skip_frames_replacement = modified_frame
                previous_frame_gray = current_frame_gray
            else:
                modified_frame = skip_frames_replacement

            out.write(modified_frame)  # save output video

            # Display the resulting frame
            cv2.imshow('modified_frame', modified_frame)

            current_frame_number = current_frame_number + 1

            # print(current_frame_gray)# increment the total number of frames read

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
