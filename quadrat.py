# Name: Anthony Jackson
# Class: COS 473, Computer Vision

# These are the import statements for needed libraries.
import numpy as np
import cv2
import os

# This function handles mouse input.
def mouse_clicker(argument1, x, y, argument2, argument3):
    global base_image, warped_points, cast_image, rescale_value
    cv2.imshow("base_image", cast_image)

    if argument1 == cv2.EVENT_LBUTTONDOWN:
        # This creates a circle on the image.
        cv2.circle(cast_image, (x, y), 3, (255, 255, 0), 5, cv2.LINE_AA)
        cv2.imshow("base_image", cast_image)
        # This pulls the point from the image and appends it.
        if len(warped_points) < 4:
            warped_points = np.append(warped_points, [((x * rescale_value), (y * rescale_value))], axis=0)
        # Once four points are chosen the window closes.
        if len(warped_points) == 4:
            cv2.destroyAllWindows()

# This function gathers point information from the image.
def get_base_points(w, h):
    bp = np.empty((0, 2), dtype=np.int32)
    bp = np.append(bp, [(0, 0)], axis=0)
    bp = np.append(bp, [(w-1, 0)], axis=0)
    bp = np.append(bp, [(w-1, h-1)], axis=0)
    bp = np.append(bp, [(0, h-1)], axis=0)
    return bp

# This is the re-scale value for 4k images, this can be changed for smaller images or increased for smaller windows.
rescale_value = 3

# This is the name of the folder in which the files are stored.
folder_name = "photos"

# This loops through the folder "q5_photos" so that they can be manipulated.
for file in os.listdir(folder_name):
    # This gets the current command line path, current photo in the directory and warped files name.
    current_photo = os.path.join(folder_name, file)
    output_name = os.path.join(folder_name, "warped_" + str(file))

    # This reads the image into a CV2 format and rotates it for the correct shape.
    base_image = cv2.imread(current_photo)

    # This set the height/weight for the image and casts them into a dimension list and resized dimensions.
    image_h, image_w = base_image.shape[:2]
    dimensions = (int(image_w), int(image_h))
    resize_dimensions = (int((image_w / rescale_value)), int((image_h / rescale_value)))

    # This calls a function which creates a numpy ist of points from the image.
    base_points = get_base_points(image_w, image_h)

    # All the images are 4k and need to be resized by a percentage.
    base_image = cv2.resize(base_image, resize_dimensions, interpolation=cv2.INTER_AREA)

    # This creates a named window for the warp.
    cv2.namedWindow("base_image", cv2.WINDOW_AUTOSIZE)
    cast_image = base_image.copy()

    # This creates a place to append points from the selected area.
    warped_points = np.empty((0, 2), dtype=np.int32)

    # This calls the mouse function that lets them choose points.
    cv2.setMouseCallback("base_image", mouse_clicker)
    cv2.waitKey(0)

    # This warps the perspective (resizes the image first).
    base_image = cv2.resize(base_image, dimensions, interpolation=cv2.INTER_AREA)

    # This calculates the homography for the area within the point destination.
    transform, status = cv2.findHomography(warped_points, base_points)

    # This warps the image and writes it.
    skew_image = cv2.warpPerspective(base_image, transform, (image_w, image_h))
    cv2.imwrite(output_name, skew_image)

    # This removes all the windows at the end.
    cv2.destroyAllWindows()
