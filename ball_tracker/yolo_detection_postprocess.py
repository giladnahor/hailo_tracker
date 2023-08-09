import numpy as np
import cv2

np.random.seed(2) # this seed yeilds nice bb colors
COLORS = np.random.randint(0, 255, size=(91, 3), dtype=np.uint8)
    
# -------------- NMS on-chip\on-cpu functions --------------- #
# from hailo_model_zoo.core.postprocessing.detection_postprocessing import YoloPostProc

def extract_detections(input, boxes, scores, classes, num_detections, threshold=0.5):   
    for i, detection in enumerate(input):
        if len(detection) == 0:
            continue
        for j in range(len(detection)):
            bbox = np.array(detection)[j][:4]
            score = np.array(detection)[j][4]
            if score < threshold:
                continue
            else:
                boxes.append(bbox)
                scores.append(score)
                classes.append(i+1) # +1 because the classes in the network start from 1
                num_detections = num_detections + 1
    return {'boxes': boxes, 
            'classes': classes, 
            'scores': scores,
            'num_detections': num_detections}

def post_nms_infer(raw_detections, input_name):
    boxes = []
    scores = []
    classes = []
    num_detections = 0
    
    detections = extract_detections(raw_detections[input_name][0], boxes, scores, classes, num_detections)
    
    return detections

# Drawing functions
def get_label(class_id):
    with open('/home/giladn/ROS/ros_dev/install/ball_tracker/share/ball_tracker/config/yolox_s_labels.json','r') as f:
        labels = eval(f.read())         
        return labels[str(class_id)]

def _draw_detection(image, d, c, color, scale_factor_x, scale_factor_y):
    """Draw box and label for 1 detection."""
    label = get_label(c)    
    ymin, xmin, ymax, xmax = d
    ymin, xmin, ymax, xmax = int(ymin * scale_factor_y), int(xmin * scale_factor_x), int(ymax * scale_factor_y), int(xmax * scale_factor_x)    
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    image = cv2.putText(image, label, (xmin + 2, ymin - 2), font, font_scale, (255, 255, 255), 1)
    return (label, image)

def draw_detections(detections, image, min_score=0.45, scale_factor_x=1, scale_factor_y=1):
    """Draws all confident detections"""
    boxes = detections['boxes']
    classes = detections['classes']
    scores = detections['scores']
    draw = image.copy()        
    if detections['num_detections'] != 0:
        for idx in range(detections['num_detections']):
            if scores[idx] >= min_score:
                color = tuple(int(c) for c in COLORS[classes[idx]])
                label, draw = _draw_detection(draw, boxes[idx], classes[idx], color, scale_factor_x, scale_factor_y)
    return draw