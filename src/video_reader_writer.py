import os
import sys

import cv2

root_folder = os.getcwd()

video_folder_path = root_folder + '\\video\\Traffic.avi'
print(video_folder_path)
pixel_row_number = 115
pixel_col_number = 500

frame_x_axis = []  # list which contains x axis which in this case is frame number or time in ms
intensity_y_axis = []  # creating list for pixel values
timestamps_x_axis = []  # list which contains timestamp at which frame is changing


def get_derivative_with_center(x_array, y_array):
    count = 1  # gives how many times derivative is applied
    diff = [0]  # by default first derivative value is 0
    while count < len(y_array) - 1:  # run loop one value less than total
        next_x = int(x_array[count + 1])  # reads value next to count in x-array(timestam_xaxis)
        next_y = int(y_array[count + 1])
        prev_x = int(x_array[count - 1])
        prev_y = int(y_array[count - 1])
        res = int((next_y - prev_y) / (next_x - prev_x))
        if res > 20:
            res = 255
        elif res < -20:
            res = 0
        else:
            res = 0
        diff.append(res)
        count = count + 1

    diff.append(0)  # by default add 0 to last derivative value
    return diff  # return the diff list to the variable where it is called


def modify_frame(frame, derivative_as_new_intensity):
    """
    we modify each frame via manipulating intensities before putting it back to video for saving
    in this case new intensity is actually derivative of that pixel.
    we hav already changed - intensity to 0 so that we can have valid intensity of frame
    """
    # new_frame = np.zeros(frame.shape, frame.dtype)  # receives frame and send back frame with all intensities as 0
    frame[pixel_row_number][pixel_col_number] = derivative_as_new_intensity
    frame[pixel_row_number][pixel_col_number + 1] = derivative_as_new_intensity
    frame[pixel_row_number][pixel_col_number - 1] = derivative_as_new_intensity
    frame[pixel_row_number + 1][pixel_col_number] = derivative_as_new_intensity
    frame[pixel_row_number - 1][pixel_col_number] = derivative_as_new_intensity
    frame[pixel_row_number + 1][pixel_col_number + 1] = derivative_as_new_intensity
    frame[pixel_row_number - 1][pixel_col_number - 1] = derivative_as_new_intensity
    frame[pixel_row_number + 1][pixel_col_number - 1] = derivative_as_new_intensity
    frame[pixel_row_number - 1][pixel_col_number + 1] = derivative_as_new_intensity

    return frame
    # return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def save_video(derivative_list):
    """
    This function saves video after receiving derivative array which need to be put as new intensity of that pixel
    """
    # Create a VideoCapture object
    video = cv2.VideoCapture(video_folder_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Unable to read video")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(root_folder + '\\output_videos\\outpy2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          (frame_width, frame_height))  # save the video with fps, width and height similar to input
    frame_number = 0
    """
     before reading each frame from video we initialize it, so that we can increase frame number after reading each next Frame
    """

    while True:
        is_frame_exists, frame = video.read()

        if is_frame_exists:
            gray = cv2.cvtColor(frame,
                                cv2.COLOR_BGR2GRAY)  # changing original frames to grayscale as input video is color

            # Write the frame into the file 'output.avi'
            # we save each frame in output video using out.write(frame)
            # modify_frame() function returns modified frame by manipulating intensities of specific pixels
            current_frame_derivative_specific_pixel = derivative_list[frame_number]
            modified_frame = modify_frame(frame,
                                          current_frame_derivative_specific_pixel)  # pass the grayscale frame and derivative of current frame number
            out.write(modified_frame)  # write the modified pixel in frame

            # Display the resulting frame
            cv2.imshow('frame', modified_frame)

            frame_number = frame_number + 1  # increment the total number of frames read

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):  # show the frames
                break

        # Break the loop
        else:
            break

        # When everything done, release the video capture and video write objects
    video.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def read_video_and_save_intensity_in_list():
    video = cv2.VideoCapture(video_folder_path)  # read video and save it as an object
    if video is None:
        print('Failed to load video:')
        sys.exit(1)
    frame_number = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # returns total number of  frame in the video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))  # returns number of frame in a second
    duration = (total_frames / fps) * 1000  # in ms
    print("total_frames:{0}, width:{1}, Height:{2}, FPS:{3}, duration:{4}".format(total_frames, width, height, fps,
                                                                                  duration))  # {0} means in format(first_variable name) and so on

    while video.isOpened():  # loop over the frames of the video, is video still readable

        is_frame_exists, current_frame = video.read()  # grab the current frame(bgr image) if frame exists there

        # check to see if we have reached the end of the video

        if is_frame_exists:
            frame_x_axis.append(frame_number)
            current_frame_time = int(video.get(cv2.CAP_PROP_POS_MSEC))  # extracting time span at which frame occurs
            timestamps_x_axis.append(current_frame_time)  # adding items in time matrix
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)  # changing original bgr frames to grayscale
            intensity_y_axis.append(gray[pixel_row_number][pixel_col_number])
            # will run till all the frames are read
        else:
            break

        frame_number = frame_number + 1  # increment the total number of frames read

    video.release()
    cv2.destroyAllWindows()  # destroy all the opened windows


def main():
    print('reading video.............')
    read_video_and_save_intensity_in_list()
    print('calculating derivative.............')
    derivative_with_center_list = get_derivative_with_center(frame_x_axis,
                                                             intensity_y_axis)  # taking function of derivative for all y-values
    print('saving video.............')
    save_video(derivative_with_center_list)  # call save_video function and pass the above derivative list
    print('total derivative: ', len(derivative_with_center_list))
    print('printing derivative.............')
    print(derivative_with_center_list)


main()  # we are calling main function as the first step of program
print('program end')
sys.exit(1)
