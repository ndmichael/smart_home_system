import cv2
import numpy as np
import pygame
import time

# Initialize Pygame for sound
pygame.mixer.init()
fire_sound = pygame.mixer.Sound("fire_sound.wav")  # Replace "fire_sound.wav" with the path to your sound file

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Set up initial frame
_, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)

# Parameters for movement detection
min_contour_area_movement = 500  # Adjust as needed
movement_detected = False

# Parameters for fire detection
lower_fire = np.array([0, 100, 100])
upper_fire = np.array([10, 255, 255])
min_contour_area_fire = 2000  # Adjust as needed
fire_detected = False
fire_detection_interval = 5  # Set the interval in seconds for subsequent fire detections
last_fire_detection_time = time.time()

while True:
    # Read a new frame
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute the absolute difference between the current frame and the first frame
    frame_delta = cv2.absdiff(first_gray, gray)

    # Threshold the delta image
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in the thresholded image (for movement detection)
    contours_movement, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Reset movement_detected flag
    movement_detected = False

    # Loop over the contours for movement detection
    for contour in contours_movement:
        # If the contour area is too small, ignore it
        if cv2.contourArea(contour) < min_contour_area_movement:
            continue

        # Draw a bounding box around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Movement detected
        if not movement_detected:
            movement_detected = True

    # Fire detection logic with a timer
    current_time = time.time()
    if current_time - last_fire_detection_time >= fire_detection_interval:
        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for the fire color
        mask_fire = cv2.inRange(hsv, lower_fire, upper_fire)

        # Find contours in the mask (for fire detection)
        contours_fire, _ = cv2.findContours(mask_fire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Reset fire_detected flag
        fire_detected = False

        # Loop over the contours for fire detection
        for contour_fire in contours_fire:
            # If the contour area is too small, ignore it
            if cv2.contourArea(contour_fire) < min_contour_area_fire:
                continue

            # Draw a bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour_fire)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # Fire detected
            if not fire_detected:
                fire_detected = True
                fire_sound.play()
                last_fire_detection_time = current_time  # Update the last detection time

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()


