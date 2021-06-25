#!/usr/bin/env python
"""OpenCV AV crash detection system"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import argparse
import time
import numpy as np
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import datetime

sys.path.append("..")

class Timer:
    """Class to measure time between codes"""
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise SystemExit(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise SystemExit(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return str(time.strftime('%Hh:%Mm:%Ss', time.gmtime(elapsed_time)))


def convert_sec(sec):
    return str(datetime.timedelta(seconds=sec))


def predict_crash(video_file_name, collision_threshhold=0.5, show_video=False):
    """Model preparation
    Path to frozen detection graph. This is the actual model that is used for the object detection."""
    MODEL_NAME = 'models'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    ## Load a (frozen) Tensorflow model into memory.
    DETECTION_GRAPH = tf.Graph()
    with DETECTION_GRAPH.as_default():
        OD_GRAPH_DEF = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            SERIALIZED_GRAPH = fid.read()
            OD_GRAPH_DEF.ParseFromString(SERIALIZED_GRAPH)
            tf.import_graph_def(OD_GRAPH_DEF, name='')

    NUM_CLASSES = 90
    LABEL_MAP = label_map_util.load_labelmap(PATH_TO_LABELS)
    CATEGORIES = label_map_util.convert_label_map_to_categories(LABEL_MAP, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    CATEGORY_INDEX = label_map_util.create_category_index(CATEGORIES)

    IMAGE_SIZE = (12, 8)
    TIMER = Timer()
    collision_detected = False
    with DETECTION_GRAPH.as_default():
        with tf.compat.v1.Session(graph=DETECTION_GRAPH) as sess:
            TIMER.start()
            vidObj = cv2.VideoCapture(video_file_name)
            fps = int(vidObj.get(cv2.CAP_PROP_FPS))

            success = 1
            while success:
                success, image = vidObj.read()
                try:
                    screen = cv2.resize(image, (800, 450))
                except Exception as catch_exp:
                    pass
                image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = DETECTION_GRAPH.get_tensor_by_name('image_tensor:0')
                boxes = DETECTION_GRAPH.get_tensor_by_name('detection_boxes:0')
                scores = DETECTION_GRAPH.get_tensor_by_name('detection_scores:0')
                classes = DETECTION_GRAPH.get_tensor_by_name('detection_classes:0')
                num_detections = DETECTION_GRAPH.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    CATEGORY_INDEX,
                    use_normalized_coordinates=True,
                    line_thickness=8)


                for i, b in enumerate(boxes[0]):
                    ##                 car                   bus                  truck
                    if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
                        if scores[0][i] >= 0.5:
                            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                            apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4), 1)
                            cv2.putText(image_np, '{}'.format(apx_distance),
                                        (int(mid_x*800), int(mid_y*450)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                            if apx_distance <= collision_threshhold:
                                if mid_x > 0.3 and mid_x < 0.7:
                                    collision_detected = True

                                    current_frame = vidObj.get(cv2.CAP_PROP_POS_FRAMES)
                                    crash_duration = int(current_frame/fps)
                                    return collision_detected, convert_sec(crash_duration)

                if show_video:
                    image_ret = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Window', cv2.resize(image_ret, (800, 450)))
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        print('Elapsed Time : ' + TIMER.stop())
                        cv2.destroyAllWindows()
                        break
            return collision_detected, ''


DEF_PARSER = argparse.ArgumentParser(description='Script to detect crash in a video')
DEF_PARSER.add_argument('--show-video','-s', dest='show_live', default=False, action='store_true', help='Show live detection')
DEF_PARSER.add_argument('--file-path', '-f', dest='file_name', default='videos/vid1.mp4', type=str, help='Provide video path.')
DEF_PARSER.add_argument('--collision-thresh', '-ct', dest='collision_threshhold', default=0.2, type=str, help='Provide collision collision.')
PARSER_OBJ = DEF_PARSER.parse_args()

if PARSER_OBJ.file_name == None:
    DEF_PARSER.print_help()
    DEF_PARSER.error('Please provide video file path')
else:
    crash_detect_flag, collision_time = predict_crash(PARSER_OBJ.file_name, collision_threshhold=PARSER_OBJ.collision_threshhold, show_video=PARSER_OBJ.show_live)
    if crash_detect_flag:
        print("Collision Detected @ " + collision_time)
    else:
        print("No Collision Detected")
