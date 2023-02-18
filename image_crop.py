import cv2
import os

# Path to the folder containing the images
folder_path = "plots"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        img = cv2.imread(os.path.join(folder_path, filename))

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a mask of the non-white areas
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

        # Find the contours of the non-white areas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour and draw it on the mask
        largest_contour = max(contours, key=cv2.contourArea)
        mask = cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # Find the bounding box of the largest non-white area
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image based on the bounding box
        cropped_img = img[y:y+h, x:x+w]

        # Save the cropped image
        cv2.imwrite(os.path.join(folder_path, filename), cropped_img)
