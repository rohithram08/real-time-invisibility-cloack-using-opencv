import cv2
import numpy as np
import time

# Attempt to open the default camera (index 0) using the default backend.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera!")
    exit()

# Allow the camera to warm up.
time.sleep(3)

print("Capturing background...")
background_frames = []
num_frames_required = 5  # Adjust this number as needed.

# Capture a set number of valid frames for the background.
while len(background_frames) < num_frames_required:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Unable to capture a frame for background; retrying...")
        time.sleep(0.1)
        continue
    background_frames.append(frame)

# Check if we captured any frames. If not, exit gracefully.
if len(background_frames) == 0:
    print("Error: No frames captured for background!")
    cap.release()
    exit()

# Compute the median background over the captured frames.
background = np.median(background_frames, axis=0).astype(np.uint8)
background = cv2.resize(background, (640, 480))
print("Background captured successfully!")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame!")
            break

        # Resize frame for consistency.
        frame = cv2.resize(frame, (640, 480))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define HSV range for the blue color.
        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([130, 255, 255])

        # Create a mask for detecting blue color.
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_inv = cv2.bitwise_not(mask)

        # Use the mask to extract the background and the non-blue portions of the frame.
        res1 = cv2.bitwise_and(background, background, mask=mask)
        res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Blend the extracted regions to achieve the invisibility cloak effect.
        final_output = cv2.addWeighted(res1, 0.7, res2, 0.3, 0)

        cv2.imshow("Invisibility Cloak", final_output)

        # Exit on pressing 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Released camera and closed all windows.")
