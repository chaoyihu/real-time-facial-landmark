# Created by Chaoyi Hu, Dec 2023

#################################
# Prepare
#################################
import cv2
import numpy as np
from datetime import datetime

import torch

from face_detector.blazeface import BlazeFace
from face_landmarks.facemesh import FaceMesh


##################################
# Create face detector
##################################

# # Option 1: front camera (close-up)
# detector = BlazeFace()
# detector.load_weights("face_detector/blazeface.pth")
# detector.load_anchors("face_detector/anchors.npy")
# detector.min_score_thresh = 0.75
# detector.min_suppression_threshold = 0.3
# desired_size = 128

# Option 2: Back camera (distant)
detector = BlazeFace(back_model=True)
detector.load_weights("face_detector/blazefaceback.pth")
detector.load_anchors("face_detector/anchorsback.npy")
detector.min_score_thresh = 0.75
detector.min_suppression_threshold = 0.3
desired_size = 256


##################################
# Create face landmarks predictor
##################################

device = 'cpu'
predictor = FaceMesh().to(device)
predictor.load_state_dict(
    torch.load(
        'face_landmarks/model_checkpoint.pth', 
        map_location=torch.device(device)
    )['model_state_dict']
)
n_points = 68


##################################
# Run inference
##################################

# Get input
cap = cv2.VideoCapture(0) # option 1: Capture the image from webcam
# cap = cv2.VideoCapture("data/test_videos/v1.mp4") # option 2: Read frame from video file

# inference
detection_time = 0
prediction_time = 0
total_time = 0
n_frames = 0

while True:
    start_time = datetime.now()
    
    # read original image
    success, image = cap.read()
    if not success:
        continue
    original_w, original_h, channels = image.shape
    
    # pad if not square
    if original_w < original_h:
        diff = original_h - original_w
        padding = diff // 2
        image = cv2.copyMakeBorder(image, padding, diff - padding, 0, 0, cv2.BORDER_CONSTANT, value=0)
    elif original_w > original_h:
        diff = original_w - original_h
        padding = diff // 2
        image = cv2.copyMakeBorder(image, 0, 0, diff, padding, cv2.BORDER_CONSTANT, value=0)
    w, h, c = image.shape

    # Detect face
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(imgRGB, (desired_size, desired_size))
    detections = detector.predict_on_image(img)
    if len(detections) == 0:
        continue
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)
    print("Found %d faces" % detections.shape[0])
 
    take_time_1 = datetime.now()
    delta = (take_time_1 - start_time).total_seconds() * 1000
    detection_time += delta

    # Predict landmarks on each face
    for i in range(detections.shape[0]):
        # Facial bounding box
        ymin = int(detections[i, 1] * h)
        xmin = int(detections[i, 1] * w)
        ymax = int(detections[i, 2] * h)
        xmax = int(detections[i, 3] * w)
        rect_l, rect_t = xmin, ymin
        rect_r, rect_b = xmax, ymax
        cv2.rectangle(image, (rect_l, rect_t), (rect_r, rect_b), (255, 0, 255), 1)
        
        # get image crop of facial area
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        span_x = xmax - xmin
        span_y = ymax - ymin
        xmin, xmax = max(0, cx - span_x), min(w, cx + span_x)
        ymin, ymax = max(0, cy - span_y), min(h, cy + span_y)
        if xmin >= xmax or ymin >= ymax:
            continue
        image_crop = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        crop_h, crop_w, channels = image_crop.shape

        # Landmarks
        input_image = cv2.resize(image_crop, (192, 192))
        input_image = torch.from_numpy(input_image).permute(2, 0, 1)
        input_image = input_image.unsqueeze(0).float().to(device)
        coord, confidence = predictor(input_image)
        coord = coord.view(3, -1)
        ratio_x = 192 / crop_w
        ratio_y = 192 / crop_h
        coord[0, :] /= ratio_x
        coord[1, :] /= ratio_y
        coord[0, :] += xmin
        coord[1, :] += ymin
        for i in range(coord.shape[1]):
            x, y, z = coord[0, i].item(), coord[1, i].item(), coord[2, i].item()
            cv2.circle(image, (int(x), int(y)), 1, (0,255,0), -1)
       
    take_time_2 = datetime.now()
    delta = (take_time_2 - take_time_1).total_seconds() * 1000
    prediction_time += delta
    
    delta = (take_time_2 - start_time).total_seconds() * 1000
    total_time += delta
    fps = 1000 / delta
    cv2.putText(image, "FPS: {}".format(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
    cv2.imshow("Facial Detection", image)
    
    n_frames += 1
    
    if cv2.waitKey(10) == 27: # press esc key to exit
        break

cap.release()

print("Number of frames processed:          {}".format(n_frames))
print("Avg time for face Detection:         {:.2f} ms per frame".format(detection_time / n_frames))
print("Avg time for landmark prediction:     {:.2f} ms per frame".format(prediction_time / n_frames))
print("Avg processing rate:                 {:.2f} frame per second (FPS)".format(1000 / (total_time / n_frames)))
