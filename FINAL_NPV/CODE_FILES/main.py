import json
import cv2
from ultralytics import YOLO
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime


os.makedirs("json", exist_ok=True)


cap = cv2.VideoCapture("/Users/dikshantthapa/Desktop/NepalPlateVision/FINAL_NPV/Nepali_Road.mp4")

model = YOLO("/Users/dikshantthapa/Desktop/NepalPlateVision/FINAL_NPV/plate_v11.pt")


count = 0


className = ["License"]


digit_model = YOLO("/Users/dikshantthapa/Desktop/NepalPlateVision/FINAL_NPV/best_Sep.pt")

def extract_characters(frame, x1, y1, x2, y2):
   
    plate_img = frame[y1:y2, x1:x2]
    
    
    results = digit_model.predict(plate_img, conf=0.45)
    characters = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = result.names[int(box.cls[0])]
            characters.append(label) 
            
    return "".join(characters)  

def save_json(license_plates, startTime, endTime):
   
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plate": list(license_plates)
    }
    interval_file_path = "json/output_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent=2)

    
    cummulative_file_path = "json/LicensePlateData.json"
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

  
    existing_data.append(interval_data)

    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)

   
    save_to_database(license_plates, startTime, endTime)

def save_to_database(license_plates, start_time, end_time):
    conn = sqlite3.connect('licensePlatesDatabase.db')
    cursor = conn.cursor()
    for plate in license_plates:
        cursor.execute('''
            INSERT INTO LicensePlates(start_time, end_time, license_plate)
            VALUES (?, ?, ?)
        ''', (start_time.isoformat(), end_time.isoformat(), plate))
    conn.commit()
    conn.close()


startTime = datetime.now()
license_plates = set()

while True:
    ret, frame = cap.read()
    if ret:
        currentTime = datetime.now()
        count += 1
        print(f"Frame Number: {count}")
        
        
        results = model.predict(frame, conf=0.45)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                
                label = extract_characters(frame, x1, y1, x2, y2)
                if label:
                    license_plates.add(label)
                
                
                textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        
        if (currentTime - startTime).seconds >= 20:
            endTime = currentTime
            save_json(license_plates, startTime, endTime)
            startTime = currentTime
            license_plates.clear()
        
        
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
