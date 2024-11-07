from face_detector import YoloDetector
import numpy as np
from PIL import Image
import cv2


model = YoloDetector(target_size=720, device="cpu", min_face=90)

# Path to video file
path_to_video = 'test_video.mp4'

# Open video file
video = cv2.VideoCapture(path_to_video)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    # Run model prediction
    bboxes, points = model.predict(frame)
    
    # Draw bounding boxes on the frame
    for bbox in bboxes[0]:  # assuming bboxes[0] contains all boxes for the frame
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Face Tracking Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
