import os
import sys

import cv2
import numpy as np

root = os.getcwd()
root_path = root + '\\video\\Traffic.avi'


def modify_frame(modified_frame_gray, current_frame_gray, current_frame_number, read_video_counter):
    height, width = modified_frame_gray.shape  # defines grayscale(2D) image height and width
    new_frame = np.copy(modified_frame_gray)
    for j in range(0, width):
        current_y = int(current_frame_gray[read_video_counter, j])
        new_frame[current_frame_number, j] = current_y
    return new_frame


def read_video_and_save():
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

    size = (total_frames, width)

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    print("total_frames:{0}, width:{1}, Height:{2}, FPS:{3}, duration:{4}".format(total_frames, width, height, fps,
                                                                                  duration))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(root + '\\video\\mod.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          (width, total_frames), 0)
    current_frame_number = 0
    read_video_counter = 0
    read_frames_time_in_ms = 100  # time at which u want to read video, it will be 100,200,300,400....

    """
     before reading each frame from video we initialize it, so that we can increase frame number after reading each next Frame
    """
    modified_frame_gray = np.zeros(size, np.uint8)
    while True:  # if statement id true, execute these set of statements

        is_frame_exists, frame = video.read()
        current_frame_time = int(video.get(cv2.CAP_PROP_POS_MSEC))  # extracting time span
        if is_frame_exists:
            current_frame_gray = cv2.cvtColor(frame,
                                              cv2.COLOR_BGR2GRAY)  # changing original frames to grayscale as input video is color

            modified_frame_gray = modify_frame(modified_frame_gray, current_frame_gray,
                                               current_frame_number, read_video_counter)
            # print(modified.shape)

            # Display the resulting frame
            current_frame_number = current_frame_number + 1
            # If the last frame is reached, reset the capture and the frame_counter
            if current_frame_number == total_frames and read_video_counter < total_frames - 1:
                cv2.imshow('new', modified_frame_gray)
                out.write(modified_frame_gray)  # save output video
                modified_frame_gray = np.zeros(size, np.uint8)
                read_video_counter = read_video_counter + 1
                current_frame_number = 0  # Or whatever as long as it is the same as next line
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print('Please wait program is running')

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
    read_video_and_save()


main()
print('program end')
sys.exit(1)
