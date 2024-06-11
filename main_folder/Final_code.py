import cv2
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO

# Load the custom YOLO model
model = YOLO(r'E:\Train_wagon_crop_image\person_count\person_Count_code\best (5).pt')

# Define the areas for detection

area2 = [(945,23), (958,23), (958,390),(945,390)]

# Function to get mouse coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video file
cap = cv2.VideoCapture(r'E:\Train_wagon_crop_image\person_count\person_Count_code\video.mp4')

# Load class names
with open(r"E:\Train_wagon_crop_image\person_count\person_Count_code\coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Create directory for saving cropped images
output_dir = r"E:\Train_wagon_crop_image\person_count\person_Count_code\cropped_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize counter and set for unique wagon IDs
count = 0
cropped_wagons = set()

# Function to generate a unique ID for each bounding box
def generate_unique_id(x1, y1, x2, y2):
    return f'{x1}-{y1}-{x2}-{y2}'

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    boxes_data = results[0].boxes.data
    boxes_df = pd.DataFrame(boxes_data).astype("float")
             
    for index, row in boxes_df.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        class_name = class_list[d]
        if 'wagon' in class_name:
            result = cv2.pointPolygonTest(np.array(area2, np.int32), ((x2, y2)), False)
            if result >= 0:
                unique_id = generate_unique_id(x1, y1, x2, y2)
                
                # Check if the unique ID is already in the cropped_wagons set
                if unique_id not in cropped_wagons:
                    cropped_wagons.add(unique_id)
                    
                    # Draw the bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (x2, y2), 5, (255, 0, 255), -1)
                    cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Crop and save the image
                    cropped_image = frame[y1:y2, x1:x2]
                    crop_filename = os.path.join(output_dir, f'cropped_wagon_{count}_{index}.jpg')
                    cv2.imwrite(crop_filename, cropped_image)
                    print(f'Image saved: {crop_filename}')
        
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('Define region'), (882, 407), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(2) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
