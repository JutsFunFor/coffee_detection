
import numpy as np
import cv2 as cv
from tflite_runtime.interpreter import Interpreter
import re
import argparse
from PIL import Image


class ModelEvalTFLite:

    def __init__(self):
        self.image = None

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


    def crop_image(self, img, config=0):
        """Cropping image according config"""
        img = np.asarray(img)

        if config:
            img = img[0:100, 0:150]
        else:
            img = img[150:250, 350:500]

        return img


    def capture(self, camera_id, camera_config):
        """Capturing the image from camera_id"""
        cap = cv.VideoCapture(camera_id)

        if cap.isOpened():
            try:
                ret, frame = cap.read()
                if ret:
                    input_image = Image.fromarray(frame).convert('RGB')  # .resize((input_width, input_height), Image.ANTIALIAS)
                    input_image = self.crop_image(input_image, camera_config)
                    return input_image

            except Exception as e:
                print(e)


    def initialize(self):

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '--model', help='File path of .tflite file.', required=True)
        parser.add_argument(
            '--labels', help='File path of labels file.', required=True)
        parser.add_argument(
            '--rstp_address',
            help='IP camera rstp address. Example - r"rtsp://admin:pipipi@192.168.104.15" ',
            required=True)
        parser.add_argument(
            '--config',
            help='Configuration of cropping images according name of complex and cup position.\n'
                 ' Example: "1_17-left or "Skolkovo-right""',
            required=True)
        parser.add_argument(
            '--threshold',
            help='Score threshold for detected objects.',
            required=False,
            type=float,
            default=0.4)

        args = parser.parse_args()

        labels = self.load_labels(args.labels)

        interpreter = Interpreter(args.model)
        interpreter.allocate_tensors()

        _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

        # Capturing image and cropping it
        # self.capture returns cropped image

        self.image = self.capture(args.rstp_address, args.config)

        if self.image is not None:
            results = self.detect_objects(interpreter, self.image, args.threshold)

            if results:
                for obj in results:
                    # Convert the bounding box figures from relative coordinates
                    # to absolute coordinates based on the original resolution
                    ymin, xmin, ymax, xmax = obj['bounding_box']
                    xmin = int(xmin * input_width)
                    xmax = int(xmax * input_width)
                    ymin = int(ymin * input_height)
                    ymax = int(ymax * input_height)

                    label = labels[obj['class_id']]
                    score = obj['score']

                    return xmin, ymin, xmax, ymax, label, score, self.image
            else:
                print('No results')
        else:
            print('No image captured')


if __name__ == '__main__':
    
    modelEval = ModelEvalTFLite()
    
    xmin, ymin, xmax, ymax, label, score, image = modelEval.initialize()
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cv.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0))
    cv.imwrite('out_1.png', image)
    print(label, score)





