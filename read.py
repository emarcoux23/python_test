import cv2 as cv
import numpy as np
import os

# Create a directory to save the cropped images if it doesn't exist
output_dir = "output_tracking_numbers"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the image
img = cv.imread("images/image1.jpg")

# Check if the image is loaded successfully
if img is None:
    raise Exception("Image not loaded.")

# Create a copy of the original image to use for ROI extraction
img_copy = img.copy()

# Convert the image to HSV color space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Define the range for yellow color in HSV
lower_yellow = np.array([20, 125, 150])
upper_yellow = np.array([40, 255, 255])

# Create a mask for yellow color
mask = cv.inRange(hsv, lower_yellow, upper_yellow)

# Find contours in the mask
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

index = 0
# Loop over the contours to draw bounding boxes
for contour in contours:
    # Get the bounding box for each contour
    x, y, w, h = cv.boundingRect(contour)
    # Check if the contour is at least 40 pixels by 10 pixels
    if w >= 40 and h >= 10:
        # Draw the bounding box on the original image
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Add the text beside the bounding box
        cv.putText(img, "Tracking number " + str(index + 1), (x + w - 175, y + h - 60 // 2), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Extract the region of interest (ROI) from the copy of the original image
        roi = img_copy[y:y+h, x:x+w]

        # Upscale the ROI using bilinear interpolation
        upscale_factor = 5  # You can change this factor as needed
        roi_upscaled = cv.resize(roi, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv.INTER_LINEAR)

        # Save the upscaled ROI as a separate image
        roi_filename = os.path.join(output_dir, f"Tracking number {index + 1}.jpg")
        cv.imwrite(roi_filename, roi_upscaled)

        index += 1

# Save the original image with bounding boxes and text
annotated_img_filename = os.path.join(output_dir, "annotated_image.jpg")
cv.imwrite(annotated_img_filename, img)

# Display the original image with bounding boxes and text
#cv.imshow("Yellow Squares", img)

# Wait for a key press indefinitely
#cv.waitKey(5000)
cv.destroyAllWindows()
