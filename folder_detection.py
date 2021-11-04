import numpy as np
import cv2 as cv
from tflite_object_detection import ModelEvalTFLite
import os


def main():
    folder = 'all_cropped/'
    model_name = 'tflite_ssd_mobilenet_100x150.tflite'
    labelmap_path = 'labelmap_100x150.pbtxt'
    threshold = 0.8
    for filename in os.listdir(folder):
        if filename.endswith('jpg') or filename.endswith('png'):
            img = cv.imread(folder + filename, cv.IMREAD_UNCHANGED)
            img = np.array(img)
            img = np.expand_dims(img, 0)

            evaluator = ModelEvalTFLite()
            xmin, ymin, xmax, ymax, labels, score = evaluator.initialize(filename, model_name, labelmap_path, threshold)

            cv.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0))
            cv.addText(img, f'{labels}-{score}', nameFont='arial', org='')
            cv.imwrite(folder + filename + 'tested', img)




if __name__ == '__main__':
    main()