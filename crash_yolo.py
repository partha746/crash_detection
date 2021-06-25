#!/usr/bin/env python
"""Yolov4 AV crash detection system"""

# download YOLOv4 weight
# https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import argparse
import imutils
import cv2
import datetime

def convert_sec(sec):
    return str(datetime.timedelta(seconds=sec))


def predict_crash(video_file_name, collision_threshhold, min_confidance, show_video, use_tiny=True):
    model_folder = 'models'

    tiny_weights = ''
    if use_tiny:
        tiny_weights = '-tiny'

    labelsPath = os.path.join(model_folder, "coco.names")
    weightsPath = os.path.join(model_folder, "yolov4" + tiny_weights + ".weights")
    configPath = os.path.join(model_folder, "yolov4" + tiny_weights + ".cfg")

    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(2)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    vs = cv2.VideoCapture(video_file_name)
    (W, H) = (None, None)
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    collision_detected = False
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > min_confidance:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_confidance, 0.3)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                apx_distance = round(1 - (boxes[i][3] - boxes[i][1]), 1)

                if apx_distance <= collision_threshhold:
                    collision_detected = True
                    current_frame = vs.get(cv2.CAP_PROP_POS_FRAMES)
                    crash_duration = int(current_frame/fps)

                    collision_detected = True

                    return collision_detected, convert_sec(crash_duration)

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.1f}% d:{:.1f}u".format(LABELS[classIDs[i]],
                    round(confidences[i]*100, 1), apx_distance)
                cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if show_video:
            cv2.imshow("output", cv2.resize(frame, (800, 600)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                   break
    vs.release()
    return collision_detected, ''


DEF_PARSER = argparse.ArgumentParser()
DEF_PARSER.add_argument('--file-path', '-f', dest='file_name', required=False, default='videos/vid1.mp4',
    help="path to input video")
DEF_PARSER.add_argument('--tiny', '-t', dest='tiny', default=False, 
    action='store_true', help='Used for tiny weights.')
DEF_PARSER.add_argument("-c", "--confidence", dest='confidence', type=float, default=0.5,
    help="minimum probability to filter weak detections")
DEF_PARSER.add_argument('--collision-thresh', '-ct', 
    dest='collision_threshold', default=80, type=str, help='Provide collision collision.')
DEF_PARSER.add_argument('--show-video','-s', dest='show_live', default=False, action='store_true', help='Show live detection')
PARSER_OBJ = DEF_PARSER.parse_args()

if PARSER_OBJ.file_name == None:
    DEF_PARSER.print_help()
    DEF_PARSER.error('Please provide video file path')
else:
    crash_detect_flag, collision_time = predict_crash(video_file_name=PARSER_OBJ.file_name, min_confidance=PARSER_OBJ.confidence, 
        collision_threshhold=PARSER_OBJ.collision_threshold, show_video=PARSER_OBJ.show_live)
    if crash_detect_flag:
        print("Collision Detected @ " + collision_time)
    else:
        print("No Collision Detected")
