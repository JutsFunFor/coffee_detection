from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2 as cv

start = time.time()
center_net_path = 'C:/Users/Admin/Desktop/tensorflow/workspace/training demo/exported-models/ssd_mobilenet_100x150'
pipeline_config = center_net_path + '/pipeline.config'
model_path = center_net_path + '/checkpoint/'
label_map_path = 'C:/Users/Admin/Desktop/tensorflow/workspace/training demo/exported-models/ssd_mobilenet_100x150/labelmap.pbtxt'
image_path = 'C:/Users/Admin/Desktop/all_cropped/Sistema CM 26.10.2021, 15-24-07(pb).jpg'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_path, 'ckpt-0')).expect_partial()


def get_model_detection_function(model):
    @tf.function
    def detect_fn(image):
        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


detect_fn = get_model_detection_function(detection_model)

label_map_path = label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

image = np.array(Image.open(image_path))

lab = cv.cvtColor(image, cv.COLOR_RGB2LAB)
channels = cv.split(lab)

clahe = cv.createCLAHE(clipLimit=11.5, tileGridSize=(1, 1))
channels[0] = clahe.apply(channels[0])
lab = cv.merge(channels)
image = cv.cvtColor(lab, cv.COLOR_LAB2RGB)

cv.imshow('image', image)
if cv.waitKey(0) == 27:
    cv.destroyAllWindows()

input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

label_id_offset = 1
image_np_with_detections = image.copy()

# Use keypoints if available in detections
keypoints, keypoint_scores = None, None
if 'detection_keypoints' in detections:
    keypoints = detections['detection_keypoints'][0].numpy()
    keypoint_scores = detections['detection_keypoint_scores'][0].numpy()


def get_keypoint_tuples(eval_config):
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'][0].numpy(),
    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
    detections['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=2,
    max_boxes_to_draw=4,
    min_score_thresh=.80,
    agnostic_mode=False,
    keypoints=keypoints,
    keypoint_scores=keypoint_scores,
    keypoint_edges=get_keypoint_tuples(configs['eval_config']))

end = time.time()

print(end-start)
plt.figure(figsize=(12, 16))
plt.imshow(image_np_with_detections)
plt.savefig('./output.png')
plt.show()
# detection_boxes = tf.squeeze(detections['detection_boxes'], [0])
# print(detection_boxes)