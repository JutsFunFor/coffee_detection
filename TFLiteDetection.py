import argparse
import io
import re
import time
from annotation import Annotator
import numpy as np
import cv2 as cv
from PIL import Image
from tflite_runtime.interpreter import Interpreter

time_start = time.time()

CAMERA_WIDTH = 150
CAMERA_HEIGHT = 100


class TFLiteDetection:

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

    def annotate_objects(self, annotator, results, labels):
        """Draws the bounding box and label for each object in the results."""
        for obj in results:
            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * CAMERA_WIDTH)
            xmax = int(xmax * CAMERA_WIDTH)
            ymin = int(ymin * CAMERA_HEIGHT)
            ymax = int(ymax * CAMERA_HEIGHT)

            # Overlay the box, label, and score on the camera preview
            annotator.bounding_box([xmin, ymin, xmax, ymax])
            annotator.text([xmin, ymin],
                           '%s\n%.2f' % (labels[obj['class_id']], obj['score']))
            return [xmin, ymin, xmax, ymax]

    def initialize(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '--model', help='File path of .tflite file.', required=True)
        parser.add_argument(
            '--labels', help='File path of labels file.', required=True)
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

        # rstp_address = 'rtsp://admin:pipipi@192.168.104.15'
        cap = cv.VideoCapture(0)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:

                # image = Image.fromarray(frame).convert('RGB').resize((input_width, input_height), Image.ANTIALIAS)
                # image = np.asarray(image)
                # print(f'Input_tensor_shape: {(input_width, input_height)}')
                image = Image.open('all_cropped/Sistema CM 26.10.2021, 15-16-49(pb).jpg').convert('RGB').resize(
                    (input_width, input_height), Image.ANTIALIAS)
                start_time = time.time()
                results = self.detect_objects(interpreter, image, args.threshold)

                for obj in results:
                    # Convert the bounding box figures from relative coordinates
                    # to absolute coordinates based on the original resolution
                    ymin, xmin, ymax, xmax = obj['bounding_box']
                    xmin = int(xmin * CAMERA_WIDTH)
                    xmax = int(xmax * CAMERA_WIDTH)
                    ymin = int(ymin * CAMERA_HEIGHT)
                    ymax = int(ymax * CAMERA_HEIGHT)

                    image = np.array(image)
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    cv.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0))

                cv.imwrite('output.png', image)
                elapsed_time = (time.time() - start_time)
                cv.imwrite('output.png', image)
                print(results)
                print(labels)
                print(f'Elapsed time: {elapsed_time}')

                cap.release()
            else:
                cap.release()
        time_end = time.time()
        res_time = time_end - time_start
        print(f'Time for all program: {res_time}')

        # with picamera.PiCamera(
        #         resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
        #     camera.start_preview()
        #     try:
        #         stream = io.BytesIO()
        #         annotator = Annotator(camera)
        #         for _ in camera.capture_continuous(
        #                 stream, format='jpeg', use_video_port=True):
        #             stream.seek(0)
        #             image = Image.open(stream).convert('RGB').resize(
        #                 (input_width, input_height), Image.ANTIALIAS)
        #             start_time = time.monotonic()
        #             results = detect_objects(interpreter, image, args.threshold)
        #             elapsed_ms = (time.monotonic() - start_time) * 1000
        #
        #             annotator.clear()
        #             annotate_objects(annotator, results, labels)
        #             annotator.text([5, 0], '%.1fms' % (elapsed_ms))
        #             annotator.update()
        #
        #             stream.seek(0)
        #             stream.truncate()
        #
        #     finally:
        #         camera.stop_preview()


if __name__ == '__main__':
    TFLiteDetection().initialize()
