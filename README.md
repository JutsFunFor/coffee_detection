# Coffee detection on Raspberry PI
## Processing images to detect coffee inside cup.

In this project I`ll try two paradighms. 

1) Coffee detection via traditional image processing. 

2) Training nerual network from TF2 model garden zoo.

So lets start from the first part.
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

I have already written the code in python represented this workflow. 
