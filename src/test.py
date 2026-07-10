#Below is a prototype Python script for a Raspberry Pi / Jetson-style system using OpenCV. It assumes the camera feed is already being delivered as normal image frames, usually BGR/RGB, not raw Bayer. This is the most practical starting point.

###############################################################################
# REAL-TIME CAMERA COLOR + INTENSITY DETECTION AND FILTERING PROTOTYPE
#
# Purpose:
# This program captures live video from a camera, analyzes each pixel for:
#   1. Color
#   2. Brightness / light intensity
#
# Then it modifies selected pixels in real time.
#
# Example use case:
#   - Detect pixels that are very bright and close to white
#   - Reduce their brightness
#   - Shift them toward a warmer yellow tone
#   - Display the modified video feed live
#
# Hardware target:
#   - Raspberry Pi 4 / 5
#   - NVIDIA Jetson Nano / Orin
#   - USB webcam or CSI camera
#
# Software required:
#   pip install opencv-python numpy
###############################################################################

import cv2
import numpy as np


###############################################################################
# USER-ADJUSTABLE SETTINGS
###############################################################################

# Camera index:
# 0 usually means the default connected camera.
# If you have multiple cameras, this may need to be changed to 1, 2, etc.
CAMERA_INDEX = 0

MAX_white_channel_threshold = 255
# Desired camera resolution.
# Lower resolution = faster processing.
# Higher resolution = more detail but slower.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Desired camera frame rate.
# 30 FPS is a good prototype target.
TARGET_FPS = 30


###############################################################################
# COLOR / INTENSITY DETECTION SETTINGS
###############################################################################

# Brightness threshold:
# Pixel brightness ranges from 0 to 255.
# 0   = black
# 255 = maximum brightness
#
# Pixels brighter than this value will be considered "too bright."
BRIGHTNESS_THRESHOLD = 150

# White detection threshold:
# A pixel is considered "white-ish" if its red, green, and blue values are all
# above this value.
#
# Example:
# R = 245, G = 240, B = 238 would be white-ish.
white_channel_threshold = 100

def change_thresh1(val):
    global white_channel_threshold
    white_channel_threshold = val

def change_thresh2(val):
    global yellow_tint_strength
    yellow_tint_strength = val / 100
# Color balance tolerance:
# True white has R, G, and B values close to each other.
#
# Example:
# R = 240, G = 238, B = 235 → white-ish
# R = 240, G = 120, B = 80  → orange/red, not white
#
# This value controls how close the R, G, and B channels must be.
WHITE_BALANCE_TOLERANCE = 35


###############################################################################
# FILTER / OUTPUT SETTINGS
###############################################################################

# Brightness reduction factor:
# 1.0 = no dimming
# 0.7 = 30% dimmer
# 0.5 = 50% dimmer
# 0.3 = 70% dimmer
DIM_FACTOR = 0.45

# Yellow tint strength:
# 0.0 = no yellow tint
# 1.0 = fully replace selected pixels with yellow target color
#
# A value between 0.2 and 0.5 usually looks more natural.
yellow_tint_strength = 0.45

# Target yellow color in BGR format.
#
# Important:
# OpenCV stores color as BGR, not RGB.
#
# BGR means:
#   Blue  first
#   Green second
#   Red   third
#
# This color is a soft warm yellow.
TARGET_YELLOW_BGR = np.array([80, 210, 255], dtype=np.uint8)


###############################################################################
# CAMERA INITIALIZATION
###############################################################################

# Open the camera.
camera = cv2.VideoCapture(CAMERA_INDEX)

# Set camera resolution.
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Set camera frame rate.
camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)

# Confirm the camera opened successfully.
if not camera.isOpened():
    raise RuntimeError("Camera could not be opened. Check camera connection/index.")


###############################################################################
# MAIN PROCESSING LOOP
###############################################################################
i = 0;

while True:

    ###########################################################################
    # STEP 1: CAPTURE ONE FRAME FROM THE CAMERA
    ###########################################################################

    # ret tells us whether the camera successfully returned a frame.
    # frame is the actual image.
    #
    # The frame is a 3D array:
    #   frame[vertical_pixel, horizontal_pixel, color_channel]
    #
    # Example:
    #   frame[100, 200] gives the pixel at row 100, column 200.
    #
    # Each pixel contains:
    #   [Blue, Green, Red]
    #
    ret, frame = camera.read()

    if not ret:
        print("Warning: Failed to capture frame.")
        continue


    ###########################################################################
    # STEP 2: SPLIT THE IMAGE INTO BLUE, GREEN, AND RED CHANNELS
    ###########################################################################

    # OpenCV uses BGR order.
    # These arrays contain the brightness of each color channel for every pixel.
    blue_channel, green_channel, red_channel = cv2.split(frame)


    ###########################################################################
    # STEP 3: CALCULATE PIXEL BRIGHTNESS / INTENSITY
    ###########################################################################

    # Human vision is more sensitive to green than red, and more sensitive to red
    # than blue. This formula estimates perceived brightness.
    #
    # Brightness = 0.114*Blue + 0.587*Green + 0.299*Red
    #
    # Result:
    #   brightness_image is a grayscale image from 0 to 255.
    #
    brightness_image = (
        0.114 * blue_channel +
        0.587 * green_channel +
        0.299 * red_channel
    ).astype(np.uint8)


    ###########################################################################
    # STEP 4: IDENTIFY PIXELS THAT ARE VERY BRIGHT
    ###########################################################################

    # bright_mask is a true/false map.
    #
    # True  = this pixel is brighter than the threshold
    # False = this pixel is not too bright
    #
    bright_mask = brightness_image > BRIGHTNESS_THRESHOLD


    ###########################################################################
    # STEP 5: IDENTIFY PIXELS THAT ARE WHITE OR CLOSE TO WHITE
    ###########################################################################

    # A white-ish pixel should have high red, green, and blue values.
    high_red = red_channel > white_channel_threshold
    high_green = green_channel > white_channel_threshold
    high_blue = blue_channel > white_channel_threshold

    # Check whether the color channels are close to each other.
    #
    # For a true white or gray pixel, R, G, and B should be similar.
    #
    red_green_close = np.abs(red_channel.astype(int) - green_channel.astype(int)) < WHITE_BALANCE_TOLERANCE
    red_blue_close = np.abs(red_channel.astype(int) - blue_channel.astype(int)) < WHITE_BALANCE_TOLERANCE
    green_blue_close = np.abs(green_channel.astype(int) - blue_channel.astype(int)) < WHITE_BALANCE_TOLERANCE

    # Combine the white tests.
    white_mask = (
        high_red &
        high_green &
        high_blue &
        red_green_close &
        red_blue_close &
        green_blue_close
    )


    ###########################################################################
    # STEP 6: CREATE FINAL MASK FOR PIXELS WE WANT TO MODIFY
    ###########################################################################

    # This final mask identifies pixels that are BOTH:
    #   1. Very bright
    #   2. White-ish
    #
    # These are the pixels we will dim and shift toward yellow.
    #
    target_mask = bright_mask & white_mask


    ###########################################################################
    # STEP 7: CREATE A COPY OF THE FRAME FOR MODIFICATION
    ###########################################################################

    # We do not directly overwrite the original frame until we calculate the
    # modified result.
    output_frame = frame.copy()


    ###########################################################################
    # STEP 8: DIM THE TARGET PIXELS
    ###########################################################################

    # Extract the target pixels from the output frame.
    target_pixels = output_frame[target_mask]

    # Reduce brightness of those pixels.
    #
    # Example:
    #   Original pixel = [240, 240, 240]
    #   DIM_FACTOR = 0.45
    #   Dimmed pixel = [108, 108, 108]
    #
    dimmed_pixels = (target_pixels.astype(np.float32) * DIM_FACTOR).astype(np.uint8)


    ###########################################################################
    # STEP 9: APPLY WARM YELLOW TINT TO TARGET PIXELS
    ###########################################################################

    # Blend the dimmed pixel with the target yellow color.
    #
    # Formula:
    #   final_pixel = dimmed_pixel * (1 - tint_strength)
    #               + yellow_pixel * tint_strength
    #
    # This avoids harsh color replacement and creates a smoother filter effect.
    #
    tinted_pixels = (
        dimmed_pixels.astype(np.float32) * (1.0 - yellow_tint_strength) +
        TARGET_YELLOW_BGR.astype(np.float32) * yellow_tint_strength
    ).astype(np.uint8)

    # Put the modified pixels back into the output image.
    output_frame[target_mask] = tinted_pixels


    ###########################################################################
    # STEP 10: OPTIONAL DEBUG VISUALIZATIONS
    ###########################################################################

    # Convert the target mask to a visible black/white image.
    #
    # White areas = pixels being modified
    # Black areas = pixels left alone
    #
    mask_display = (target_mask.astype(np.uint8) * 255)


    ###########################################################################
    # STEP 11: DISPLAY RESULTS
    ###########################################################################

    cv2.imshow("Original Camera Feed", frame)
    cv2.imshow("Brightness / White Detection Mask", mask_display)
    cv2.imshow("Filtered Output Feed", output_frame)

    if i == 0:
        cv2.createTrackbar("Brightness", "Filtered Output Feed", 0, MAX_white_channel_threshold, change_thresh1)
        cv2.createTrackbar("Tint STrenght", "Filtered Output Feed", 0, 500, change_thresh2)
        i += 1

    ###########################################################################
    # STEP 12: EXIT CONDITION
    ###########################################################################

    # Press the 'q' key to quit.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


###############################################################################
# CLEANUP
###############################################################################

camera.release()
cv2.destroyAllWindows()
