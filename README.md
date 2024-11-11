# Yolov5-Face Tracking

## Description
The project is a wrap over [yoloface](https://github.com/elyha7/yoloface.git) repo. 

## Usage example
Run test.py using your own video or webcam.

Run ``uvicorn get_bbs:app --reload`` to return bounding boxes in a frame.

To test the get_frame_bboxes endpoint, you can use:
### Browser or REST Client (e.g., Postman or Insomnia):

* Make a GET request to ``http://127.0.0.1:8000/get_frame_bboxes``.

## Citiation
Thanks [deepcam-cn](https://github.com/deepcam-cn/yolov5-face) for pretrained models and [yoloface](https://github.com/elyha7/yoloface.git) for their contribution.
