import os
import sys

import cv2
import numpy as np

root = os.getcwd()
root_path = root + '\\input_videos\\Traffic.avi'


def modify_row(modified_frame_gray, current_frame_gray, current_frame_number, i):
    height, width = modified_frame_gray.shape  # defines grayscale(2D) image height and width
    new_frame = np.copy(modified_frame_gray)  # make copy of modified frame
    for j in range(0, width):  # reads all the column values in the ith row
        intensity = int(current_frame_gray[i, j])  # gives column value of ith row
        new_frame[
            current_frame_number, j] = intensity  # put above value in frame number(row) and its column position e.g 0,0 0,1, 0,2.....
    return new_frame  # gives complete first row of first frame in first return then fst row of 2nd frame in second return


def read_video_and_save_new_frame(i):  # read each frame and apply functions
    previous_frame_gray = None
    # Create a VideoCapture object
    video = cv2.VideoCapture(root_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Unable to read video")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    duration = (total_frames / fps) * 1000  # in ms

    size = (total_frames, width)
    current_frame_number = 0
    modified_frame_gray = np.zeros(size, np.uint8)  # create an frame(h=115, w=width) and put all zeros initially
    while True:  # if statement id true, execute these set of statements

        is_frame_exists, frame = video.read()
        if is_frame_exists:
            current_frame_gray = cv2.cvtColor(frame,
                                              cv2.COLOR_BGR2GRAY)  # changing original frames to grayscale as input video is color

            modified_frame_gray = modify_row(modified_frame_gray, current_frame_gray, current_frame_number,
                                             i)  # gives new frame
            # cv2.imshow('new', modified_frame_gray)
            current_frame_number = current_frame_number + 1
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

        # When everything done, release the video capture and video write objects
    video.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    return modified_frame_gray  # gives new frame with all first rows in first return then complete 2nd frame with the 2nd rows of video


def main():
    cap = cv2.VideoCapture(root_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Unable to read video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    out = cv2.VideoWriter(root + '\\input_videos\\mod456.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          (width, total_frames), 0)  # make template for output video
    for i in range(0, height):
        mod = read_video_and_save_new_frame(i)
        # print(mod)
        out.write(mod)  # creating video
        cv2.imshow('new', mod)
        print(i, 'frame done, remaining frames are', height - i)
    out.release()


main()
print('program end')
sys.exit(1)
