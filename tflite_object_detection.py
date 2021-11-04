
import numpy as np
import cv2 as cv

from tflite_runtime.interpreter import Interpreter

import re


class ModelEvalTFLite:

    def load_labels(self, path):
        """Loads the labels file. Supports files with or without index numbers."""
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            labels = {}
            for row_number, content in enumerate(lines):
                pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
                if len(pair) == 2 and pair[0].strip().isdigit():
                    labels[int(pair[0])] = pair[1].strip()
                else:
                    labels[row_number] = pair[0].strip()
        return labels

    def set_input_tensor(self, interpreter, image):
        """Sets the input tensor."""
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, interpreter, index):
        """Returns the output tensor at the given index."""
        output_details = interpreter.get_output_details()[index]
        tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
        return tensor

    def detect_objects(self, interpreter, image, threshold):
        """Returns a list of detection results, each a dictionary of object info."""
        self.set_input_tensor(interpreter, image)
        interpreter.invoke()

        # Get all output details
        boxes = self.get_output_tensor(interpreter, 0)
        classes = self.get_output_tensor(interpreter, 1)
        scores = self.get_output_tensor(interpreter, 2)
        count = int(self.get_output_tensor(interpreter, 3))

        results = []
        for i in range(count):
            if scores[i] >= threshold:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                results.append(result)
        return results

    def initialize(self, image, model, labels, threshold=0.6):

        labels = self.load_labels(labels)

        interpreter = Interpreter(model)
        interpreter.allocate_tensors()

        _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

        results = self.detect_objects(interpreter, image, threshold)

        for obj in results:
            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * input_width)
            xmax = int(xmax * input_width)
            ymin = int(ymin * input_height)
            ymax = int(ymax * input_height)

            label = labels[obj['labels']]
            score = obj['score']

            return xmin, ymin, xmax, ymax, label, score


if __name__ == '__main__':
    image_path = 'all_cropped/Sistema CM 26.10.2021, 15-16-06(pb).jpg'
    model_path = 'tflite_ssd_mobilenet_100x150.tflite'
    labels = 'labelmap_100x150.pbtxt'
    threshold = 0.8

    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    modelEval = ModelEvalTFLite().initialize(image, model_path, labels, threshold)





