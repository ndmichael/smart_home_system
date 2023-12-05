import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

layer_names = net.getUnconnectedOutLayersNames()

# Open video capture
cap = cv2.VideoCapture(0)  # You can replace 0 with your camera index or video file path

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    height, width, _ = frame.shape

    # Detect objects using YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Process each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Fire and smoke classes in COCO dataset
                if class_id == 52 or class_id == 74:
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(detection[0] * width), int(detection[1] * height)),
                                  (int(detection[2] * width), int(detection[3] * height)), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Fire and Smoke Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
