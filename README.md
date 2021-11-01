# Coffee detection on Raspberry PI
## Processing images to detect coffee inside cup.

In this project I`ll try two paradighms. 

1) Coffee detection via traditional image processing. 

2) Training nerual network from TF2 model garden zoo.

So lets start from the first part.
## 1) Classical image processing

In order to find out the coffee lets me show some examples of images I took on my camera. There are dozen of images, so I`ll show you just couple of them.

![atl text](https://user-images.githubusercontent.com/43553016/139642156-17a152f4-a5e6-47e9-b31a-e3f0920cc558.png)

And this one

![alt text](https://user-images.githubusercontent.com/43553016/139642579-5ff150b4-ab38-4d52-8de8-d668a340bc70.jpg)

Okay, now we should determine the workflow for image processing. According the prior information about location of the cup, we can easily crop image to define region of interests (ROI) for both left and right cup. As we have several robotic coffee machines we have to set cropping parameters manualy for each machine. After this procedure we can apply diferrent filters and threshold to figure out coffee inside cup. 

So workflow includes:
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

