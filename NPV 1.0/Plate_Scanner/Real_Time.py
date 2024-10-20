# import cv2
# import numpy as np

# # Function to detect red and white regions in an image
# def detect_red_and_white(image):
#     # Convert the image to the HSV color space
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     # Define lower and upper bounds for red and white colors
#     lower_red = np.array([0, 100, 100])
#     upper_red = np.array([10, 255, 255])
#     lower_white = np.array([0, 0, 200])
#     upper_white = np.array([255, 30, 255])

#     # Create masks to isolate red and white regions
#     mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
#     mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

#     # Combine the masks to get regions that are red or white
#     combined_mask = cv2.bitwise_or(mask_red, mask_white)

#     # Apply the mask to the original image to isolate the red and white regions
#     result = cv2.bitwise_and(image, image, mask=combined_mask)

#     return result

# # Initialize the video capture (you can change the camera index if needed)
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Error reading frame")
#         break

#     # Detect red and white regions in the frame
#     red_and_white_regions = detect_red_and_white(frame)

#     # Perform additional processing and license plate detection on red_and_white_regions
#     # Implement your license plate detection logic here

#     # Display the original frame with license plate detection (for testing)
#     cv2.imshow("License Plate Detection", red_and_white_regions)

#     if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
#         break

# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # Function to filter red and white regions in an image
# def filter_red_and_white(image):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Define lower and upper bounds for red and white regions
#     lower_red = np.array([0, 0, 100])
#     upper_red = np.array([100, 100, 255])
#     lower_white = np.array([200, 200, 200])
#     upper_white = np.array([255, 255, 255])

#     # Create masks for red and white regions
#     mask_red = cv2.inRange(image, lower_red, upper_red)
#     mask_white = cv2.inRange(image, lower_white, upper_white)

#     # Combine the masks to isolate red and white regions
#     combined_mask = cv2.bitwise_or(mask_red, mask_white)

#     return combined_mask

# # Initialize the video capture (you can change the camera index if needed)
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         print("Error reading frame")
#         break

#     # Filter red and white regions in the frame
#     filtered_image = filter_red_and_white(frame)

#     # Find contours in the filtered image
#     contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Filter for rectangular contours (adjust as needed)
#     min_contour_width = 80
#     min_contour_height = 20
#     plates = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w >= min_contour_width and h >= min_contour_height:
#             plates.append(contour)

#     # Draw bounding rectangles around the license plates
#     for plate in plates:
#         x, y, w, h = cv2.boundingRect(plate)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Display the original frame with license plate contours
#     cv2.imshow("License Plate Detection", frame)

#     if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
#         break

# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()









import cv2
import numpy as np

# Function to filter red and white regions in an image
def filter_red_and_white(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for red and white regions
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])

    # Create masks for red and white regions
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

    # Combine the masks to get regions that are red or white
    combined_mask = cv2.bitwise_or(mask_red, mask_white)

    return combined_mask

# Initialize the video capture (you can change the camera index if needed)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame")
        break

    # Filter red and white regions in the frame
    filtered_image = filter_red_and_white(frame)

    # Find contours in the filtered image
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for rectangular contours based on aspect ratio
    min_aspect_ratio = 2  # Adjust this value as needed
    max_aspect_ratio = 6  # Adjust this value as needed
    plates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio >= min_aspect_ratio and aspect_ratio <= max_aspect_ratio:
            plates.append(contour)

    # Draw bounding rectangles around the license plates
    for plate in plates:
        x, y, w, h = cv2.boundingRect(plate)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original frame with license plate contours
    cv2.imshow("License Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
