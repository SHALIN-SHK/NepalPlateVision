import cv2
import numpy as np
import os

def detect_and_save(image, output_folder, net, classes):
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'car':  # Change 'car' to the relevant class for license plates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        roi = image[y:y + h, x:x + w]
        output_path = os.path.join(output_folder, f"detected_object_{i + 1}.png")
        cv2.imwrite(output_path, roi)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

if __name__ == "__main__":
    net = cv2.dnn.readNet("yolov8.cfg" , "best.pt")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getUnconnectedOutLayersNames()

    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam. Change accordingly for other video sources.

    output_folder = "detected_objects"
    os.makedirs(output_folder, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        processed_frame = detect_and_save(frame, output_folder, net, classes)

        cv2.imshow("Real-time Object Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
