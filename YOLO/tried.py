import cv2
import numpy as np
import easyocr  # Or your preferred OCR library

# Load YOLO models directly from weights (replace with your model paths)
license_plate_model = cv2.dnn.readNet("/Users/dikshantthapa/Desktop/NepalPlateVision/YOLO/best_plate.pt")
character_recognition_model = cv2.dnn.readNet("best_Sep.pt")

# Function to detect license plates, extract characters, and display results
def process_image(image):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    license_plate_model.setInput(blob)
    layerOutputs = license_plate_model.forward(license_plate_model.getUnconnectedOutLayersNames())
    bounding_boxes, class_ids, confidences = process_outputs(layerOutputs)  # Process model outputs

    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plate_image = image[y:y + h, x:x + w]
        characters = extract_characters(character_recognition_model, plate_image)
        text = easyocr.readtext(plate_image)[0][-2]  # Extract recognized text
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Process an image
image = cv2.imread("image.jpg")
processed_image = process_image(image)
cv2.imshow("Output", processed_image)
cv2.waitKey(0)

# Process a video (similar steps)