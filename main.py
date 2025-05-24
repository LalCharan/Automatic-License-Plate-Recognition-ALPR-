import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import util  # Make sure util.py has get_outputs() and NMS() methods

# Define paths to pre-trained model
model_cfg_path = os.path.join('.', 'model', 'cfg', 'yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'yolov3.weights')
class_names_path = os.path.join('.', 'model', 'coco.names')

input_dir = 'C:/Users/lalch/Downloads/automatic number plate recogniton/automatic number plate recogniton/data'

# Load class names
with open(class_names_path, 'r') as f:
    class_names = [line.strip() for line in f if line.strip()]

# Load pre-trained YOLOv3 model
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

# Create EasyOCR reader once
reader = easyocr.Reader(['en'])

# Process each image
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Failed to load image: {img_path}")
        continue

    H, W, _ = img.shape

    # Create blob from image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)
    net.setInput(blob)

    # Get detections
    detections = util.get_outputs(net)

    bboxes, class_ids, scores = [], [], []

    for detection in detections:
        bbox = detection[:4]
        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        if score > 0.5:  # Filter low confidence
            bboxes.append(bbox)
            class_ids.append(class_id)
            scores.append(score)

    # Apply Non-Maximum Suppression
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    for idx, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox
        x1, y1 = int(xc - w / 2), int(yc - h / 2)
        x2, y2 = int(xc + w / 2), int(yc + h / 2)

        license_plate = img[y1:y2, x1:x2].copy()

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # OCR Processing
        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

        output = reader.readtext(license_plate_thresh)

        for out in output:
            text_bbox, text, text_score = out
            if text_score > 0.4:
                print(f"[{img_name}] Plate: {text} (Confidence: {text_score:.2f})")

    # Visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(img_name)
    plt.axis('off')
    plt.show()
