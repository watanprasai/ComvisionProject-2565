import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
ignore_mask_color = 255

# Open the video file


file_path = filedialog.askopenfilename(title="Choose a video file", filetypes=[("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")])

cap = cv2.VideoCapture(file_path)
# Set the frame rate
cap.set(cv2.CAP_PROP_FPS, 30)

while(cap.isOpened()):
    # Read a frame from the video
    ret, frame = cap.read()
    
    if ret:
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        vertices = np.array([[(0, frame.shape[0]), (450, 310), (800, 310), (frame.shape[1], frame.shape[0])]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is not None:
            # Group the lines based on their slope and y-intercept
            line_groups = {}
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1)
                y_intercept = y1 - slope * x1
                if slope < 0:
                    # Lines with negative slope belong to the left lane
                    line_groups.setdefault('left', []).append((slope, y_intercept))
                else:
                    # Lines with positive slope belong to the right lane
                    line_groups.setdefault('right', []).append((slope, y_intercept))

            # Merge the lines in each group into a single line
            for group, lines in line_groups.items():
                if len(lines) > 0:
                    slopes, y_intercepts = zip(*lines)
                    avg_slope = np.mean(slopes)
                    avg_y_intercept = np.mean(y_intercepts)
                    x1 = int((frame.shape[0] - avg_y_intercept) / avg_slope)
                    x2 = int((310 - avg_y_intercept) / avg_slope)
                    cv2.line(frame, (x1, frame.shape[0]), (x2, 310), (0, 255, 0), 8)

        # Display the frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
