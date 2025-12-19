import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder
import cv2
import numpy as np

# **1. Load Model & Config**
PATH_TO_MODEL = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"
PATH_TO_LABELS = "label_map.pbtxt"
PATH_TO_CONFIG = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config"

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

# Load saved model
detection_model = tf.saved_model.load(PATH_TO_MODEL)

# **2. Detection Function**
def detect_objects(image_np, model):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    detections = model(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() 
                 for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    return detections

# **3. Run Detection on Test Image**
def run_detection(image_path):
    image = cv2.imread(image_path)
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    detections = detect_objects(image_np, detection_model)
    
    # Visualize results
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    scores = detections['detection_scores']
    
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            (h, w, _) = image.shape
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{category_index[classes[i]]['name']}: {int(scores[i]*100)}%"
            cv2.putText(image, label, (xmin, ymin-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)

# **4. Execute Detection**
run_detection("test_image.jpg")