# Raspberry PI coffee detection
## Processing images to detect coffee inside cup.

In this project I`ll try two paradighms. 

1) Coffee detection via traditional image processing. 

2) Training nerual network (NN) from TF2 detection model zoo.


## 1) Classical image processing

In order to find out the coffee, let me show some examples of images I took on my camera. There are dozen of images, so I`ll show you just couple of them.

![alt text](https://user-images.githubusercontent.com/43553016/139650066-7aea0794-649c-49f7-aaa7-f6e3008da5ec.jpg)


And this one

![alt text](https://user-images.githubusercontent.com/43553016/139650593-2461126d-3085-41c1-b6b5-fcf72d913112.jpg)


Okay, now we should determine the workflow for image processing. According the prior information about location of the cup, we can easily crop image to define region of interests (ROI) for both left and right cup. As we have several robotic coffee machines we have to set cropping parameters manualy for each machine. After this procedure we can apply diferrent filters and threshold to figure out coffee inside cup. 

Workflow includes:
1) Selecting coffee robotic machine
2) Cropping images according camera location inside the machine for both left and right cup.
3) Converting to greyscale
4) Applying CLAHE transformation
5) Applying threshold
6) MORPH transformation
7) Canny edge detection
8) Finding connected components 
9) Filtering result 
10) Drawing rectangle over result

I have already written function in python represented this workflow (img_processing.py). 

<img width="598" alt="Снимок экрана 2021-11-01 в 14 00 23" src="https://user-images.githubusercontent.com/43553016/139661863-d15f0640-c60e-49da-9754-08307d5a66c4.png">

Changing parameters inside code you can optimize algorithm for different light conditions and type of coffee!

## 2) Training NN

As far as we are going to use this algoritm on Raspberry PI, we should care about inference time. According this I chose ssd_mobilenet from TF2 Detection Model Zoo. Model perfect suits for cases where inference time is main criterion. Deploying a model means such stages as preparing dataset, training model, evaluating model on unseen data, quantizing weights and translating to Raspberry PI.  

Generally, I can highlight workflow as follow:

1) Collecting images for train, validation and test datasets.
2) Labeling images (CVAT.org)
3) Download pretrained NN.
4) Change config file and train NN.
5) Evaluate model.
6) Quantize weights.
7) Deploy to Raspberry PI

#### 1) Collecting images

The main part of each training process is to collect and standartize data. As for me, training dataset contains 184 images from diffferent coffee machines, light conditions and coffee types. All images was cropped according region of interests. Each image has shape of (100, 150, 3). Input tensor of ssd_mobilenet has shape of (1, 100, 150, 3). Validataion dataset includes about 50 images and test dataset has the same size.

#### 2) Labeling images

Now we have to label each image to provide tfrecords for deep learning model. I used CVAT.org for labling images. 

![image](https://user-images.githubusercontent.com/43553016/139851714-bfec2dce-b24e-4545-8503-054ed968da75.png)

After labeling you can download annotations as tfrecords and use them for training model.

####  3) Downloading pretrained network

You can easily choose model that suits your preferances at  [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md#tensorflow-2-detection-model-zoo)
I took ssd_mobilenet for fastest inference on Raspberry PI.

#### 4) Changing config file and training 

All training process described in this [tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html).

In order to provide training on small-sized images, we have to add `pad_to_multiple: 32` in the config file ![image](https://user-images.githubusercontent.com/43553016/140097574-18f6f96c-bfe5-46a5-878a-446300979604.png)


#### 5) Evaluating model 

#### 6) Quantizing weights 

After importing model, we want to run it on Raspberry PI. It is possible to reduce model size by quantizing it`s weights. 
Code make default optimizing and saves weights in float16 prec

`import tensorflow as tf`

`converter = tf.lite.TFLiteConverter.from_saved_model('tflite/saved_model/)`

`converter.optimizations = [tf.lite.Optimize.DEFAULT]`

`tflite_model = converter.convert()`

`with open('model_optimize2.tflite', 'wb') as f:` 

|
 ---- `f.write(tflite_model)`


#### 7) Deploying to Rasberry PI

Using tflite interpreter we can handle inference on Raspberry  PI. Check code!

