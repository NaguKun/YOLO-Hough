import numpy as np
import cv2 as cv

cap = cv.VideoCapture("video2.mp4")
resolution_x = 640
resolution_y = 360

# Function to calculate line coefficients
def lineCoefficient(xi, yi, xf, yf):
    if xf - xi != 0:
        a = (yf - yi) / (xf - xi)
    else:
        a = float('inf')  # Slope is infinite for a vertical line

    if a != float('inf'):
        b = yi - a * xi
    return a, b

# Detect gate using Hough lines
ret, firstFrame = cap.read()
if not ret or firstFrame is None:
    print("Error: Could not read the first frame.")
    exit()
firstFrame = cv.resize(firstFrame, (resolution_x, resolution_y))
gray_frame = cv.cvtColor(firstFrame, cv.COLOR_BGR2GRAY)
blur_frame = cv.GaussianBlur(gray_frame, (5, 5), 1.4)
edge_frame = cv.Canny(blur_frame, 150, 150)

lines = cv.HoughLinesP(edge_frame, 1, np.pi / 180, 100, None, 400, 250)
if lines is None:
    print("No lines detected.")
    exit()

edge_lines = []
for i in range(len(lines)):
    l = lines[i][0]
    x1, y1, x2, y2 = l
    cv.line(firstFrame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv.LINE_AA)
    edge_lines.append(lineCoefficient(x1, y1, x2, y2))

print(f"Edge lines: {edge_lines}")

# Initialize background subtractor
back_sub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Variables to count people
enter_count = 0
exit_count = 0

# Tracking parameters
trackers = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (resolution_x, resolution_y))
    fg_mask = back_sub.apply(frame)

    # Remove shadows (gray areas)
    _, fg_mask = cv.threshold(fg_mask, 250, 255, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    new_trackers = []
    for contour in contours:
        if cv.contourArea(contour) > 500:  # Filter small contours
            x, y, w, h = cv.boundingRect(contour)
            center = (x + w // 2, y + h // 2)

            # Update trackers
            matched = False
            for tracker in trackers:
                if cv.norm(tracker['position'], center) < 50:
                    tracker['position'] = center
                    tracker['age'] = 0
                    new_trackers.append(tracker)
                    matched = True
                    break

            if not matched:
                new_trackers.append({'position': center, 'age': 0, 'crossed': False})

    trackers = new_trackers

    # Remove old trackers
    trackers = [tracker for tracker in trackers if tracker['age'] < 30]

    # Check crossing
    for tracker in trackers:
        if not tracker['crossed']:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_coeff = lineCoefficient(x1, y1, x2, y2)
                if line_coeff != (float('inf'),):
                    a, b = line_coeff
                    # Calculate y position on the line
                    y_line = int(a * tracker['position'][0] + b)
                    if abs(tracker['position'][1] - y_line) < 5:  # Allow some margin
                        if tracker['position'][1] < y_line:
                            enter_count += 1
                        else:
                            exit_count += 1
                        tracker['crossed'] = True
                        break

    # Draw the Hough lines and display counts
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv.LINE_AA)
        cv.line(fg_mask, (x1, y1), (x2, y2), (255, 255, 255), 2, cv.LINE_AA)

    # Display counts on the frame
    cv.putText(frame, f"Enter: {enter_count}", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(frame, f"Exit: {exit_count}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frames
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fg_mask)
    
    # Exit on ESC key
    if cv.waitKey(30) == 27:
        break

cap.release()
cv.destroyAllWindows()
