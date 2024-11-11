from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
from face_detector import YoloDetector
import cv2

app = FastAPI()

# Initialize the face detector
model = YoloDetector(target_size=720, device="cpu", min_face=90)
path_to_video = 'test_video.mp4'
video = cv2.VideoCapture(path_to_video)

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class FrameResponse(BaseModel):
    frame_id: int
    bounding_boxes: List[BoundingBox]

@app.get("/get_frame_bboxes", response_model=FrameResponse)
async def get_frame_bboxes():
    if not video.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video file.")
    
    ret, frame = video.read()
    if not ret:
        raise HTTPException(status_code=404, detail="End of video reached.")

    # Run model prediction
    bboxes, points = model.predict(frame)
    
    # Extract bounding boxes in the expected format
    bounding_boxes = [
        BoundingBox(x1=int(bbox[0]), y1=int(bbox[1]), x2=int(bbox[2]), y2=int(bbox[3]))
        for bbox in bboxes[0]
    ]
    
    # Get the current frame position
    frame_id = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    
    return FrameResponse(frame_id=frame_id, bounding_boxes=bounding_boxes)

@app.on_event("shutdown")
def release_resources():
    # Release video capture when the app shuts down
    video.release()
