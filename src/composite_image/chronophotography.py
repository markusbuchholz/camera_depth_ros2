import cv2 as cv
import argparse
import os
from os import path as path

# Command line arguments
parser = argparse.ArgumentParser(description="Generate a chronophotography based on a folder of images or a video")
parser.add_argument("--inputpath", required=True, type=str, help="The path to the input video or folder containing images")
parser.add_argument("--filetype", type=str, default="png", help="Extension if input is image sequence")
parser.add_argument("--time", type=float, help="The time in seconds between each of the frames that should be used to generate the final image, if the input is a video")
parser.add_argument("--framerate", type=int, default=30, help="The framerate in frames per second (FPS) if the input is a video")
parser.add_argument("--nameformat", type=str, default="frame_", help="Provide the format used for the names of the pictures if you're using an image sequence as input. (e.g. \"frame_\"")
parser.add_argument("--show", action="store_true", help="Show the final image in an opencv window")
args = parser.parse_args()

# Input path is inexisting
if not path.exists(args.inputpath):
    print("Input path does not exist\nExiting program")
    exit()

def difference_on_base(base, current_final, current_frame):
    diff = cv.cvtColor(cv.subtract(base, current_frame), cv.COLOR_BGR2GRAY) # Difference between the base image and the current frame
    _, epic_inverse_mask = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV) # Everything which is not black becomes black and black becomes white
    different_element = cv.bitwise_and(current_frame, current_frame, mask=diff) # We take the different pixels in the current frame
    bg = cv.bitwise_and(current_final, current_final, mask=epic_inverse_mask) # The place where the different element should be pasted becomes black
    result = cv.add(bg, different_element) # Selected pixels are pasted onto current_final
    return result

if path.isdir(args.inputpath): # Input path points to a folder

    # Defining some variables
    frame_path_base = path.join(args.inputpath, args.nameformat)
    first_frame = cv.imread(frame_path_base+"0."+args.filetype)
    final_image = first_frame
    number_of_frames = 0

    # Number of files in the input directory that follow the provided name format
    for filename in os.listdir(args.inputpath):
        if path.basename(filename).startswith(args.nameformat):
            number_of_frames += 1
    
    # No valid files in the input directory
    if number_of_frames < 1:
        print("Input directory does not contain files that match the provided name format\nExiting program")
        exit()

    # Adding difference between every frame and the base image to the final image
    for i in range(1, number_of_frames):
        current_frame = cv.imread(frame_path_base+str(i)+"."+args.filetype)
        final_image = difference_on_base(first_frame, final_image, current_frame)

else:

    # Check if the input file is a video
    try:
        video = cv.VideoCapture(args.inputpath)
    except:
        print("Input path is not a video or a folder")
        exit()
    
    frame_number = 0

    # User specifies the amount of time between two of the used images
    if args.time:
        every_frame = round(args.time * args.framerate)
    else:
        every_frame = 1

    # Read frames from the input file
    while True:
        success, current_frame = video.read()
        if not success:
            break
        
        # First frame
        if frame_number == 0:
            first_frame = current_frame
            final_image = first_frame
            frame_number += 1
            continue
        
        # Check if the frame number is ok 
        if frame_number%every_frame == 0:
            final_image = difference_on_base(first_frame, final_image, current_frame)
        frame_number += 1

# Save the final image
cv.imwrite(path.join(args.inputpath, "output_img."+args.filetype), final_image)

# The user wants to see the final image in an opencv window
if args.show:
    cv.imshow("Output image", final_image)
    cv.waitKey(0)
    cv.destroyAllWindows()