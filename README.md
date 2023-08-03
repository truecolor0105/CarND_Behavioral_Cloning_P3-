# Behavioral Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Files Submitted
---

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

- [<b>model.py</b> - The script used to create and train the model](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/model.py)
- [<b>drive.py</b> - The script to drive the car (<b>IMPORTANT: model.py must be present in same folder!</b>)](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/drive.py)
- [<b>model.h5</b> - The saved model](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/model.h5)
- [<b>README.md</b> - A summary of the project](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/README.md)
- [<b>videoTrack1.mp4</b> - A video recording of driving autonomously around track1](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/videoTrack1.mp4)

    ![track1](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/images/videoTrack1-r10.gif?raw=true)


- [<b>videoTrack2.mp4</b> - A video recording of driving autonomously around track2](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/videoTrack2.mp4)

    ![track2](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/images/videoTrack2-r10.gif?raw=true)

  
---

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

<b>IMPORTANT:</b> model.py must be present in same folder, because drive.py imports a hyperparameter for clahe transform

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



Model Architecture and Training Strategy
---

#### 1. An appropriate model architecture has been employed

My model (model.py lines 321-360) is almost identical to the [Nvidia convolution neural network for self driving cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

The only changes I made are:

- The input layer is for grayscale images (1 channel instead of 3).
- After the input layer, the images are cropped.
- After the cropping layer, the data is normalized.
- In between the last convolutional layer and flatten layer, I inserted a dropout layer.

A visualization of the Network Architecture is created, showing that there are 14 layers. For each layer, the input and output shapes are given. (model.py - line 369):

![track2](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/images/model.gif?raw=true)

- The input layer has 1 channel, expecting gray scale images of 160x320x1.
- The 2nd layer crops the images in vertical direction, to eliminate the hood of the car and non-relevant sections above the road. The image size is reduced to 65x320x1.
- The 3rd layer is a Lambda layer, normalizing the data.
- Layers 4-8 are Convolution layers, to incrementally detect even more complex features in the images. 
- Layer 9 is a Dropout layer, which was added to avoid over-fitting. It is shown below that without this layer, the model is prone to over-fitting, but that by adding just this layer results in a very nicely converging solution.
- Layer 10 flattens the data.
- Layer 11-14 are Densely connected layers (= activation(dot(input, kernel) + bias)), using relu activation.
- Layer 14 is also the output layer, giving the predicted steering angle as output.



#### 2. How I got it to work - grayscale, clahe, dropout, lots-of-data

While developing the model I had some spectacular virtual crashes. I will explain how I got it to work by discussing this stubborn failure on track 2 that was the very last one to resolve:

![track2](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/images/videoTrack2-crash.gif?raw=true) <b>Crash on a difficult section of Track 2</b> 


This section is particularly challenging, because:

- it contains a very bright section (sun) next to a very dark section (shadow)
- the change occurs in the middle of a sharp turn

The issue with the contrast is addressed by applying grayscale and clahe transforms  during image generation (model.py - generator - line 199), followed by a cropping layer in the keras model (model.py - line 337):

![track2](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/images/track2-crash-image-processing.gif?raw=true)

Up to this point in my model development, I had created training data sets for track 1 and track 2, driving the car in the center of the road, in both directions. This was a lot of data already, but still the car crashed. My observation was that the "autonomous driver" lost track of the center line and then did not know what to do. Reviewing the training data, I realized that this type of situation was not yet covered and the model was not trained how to respond.

I created 4 additional training data sets. For track 1 & track 2, I drove while 'hugging the line'. I did this for the right line and the left line.
Here I am showing the training data created for track2, hugging the left line:

![track2](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/images/track2-hug-left-line.gif?raw=true) <b>Training data, using <i>'left line hugging'</i> drive style</b>


I re-trained the model, and tried it out on both tracks. It perfectly handled track 1, but track 2 still crashed in the same location. I scratched my head a few times, and then few times more, trying to figure out how to fix this. I was sure I had sufficient data, convergence was good (see below), the model was a proven CNN, so what could it be ? I then decided to use a similar trick as what was done for the left and right camera images. 

For all the images created during 'line hugging' drive style, I applied a small correction of 0.2 to the steering angle to nudge the car to move towards the center of the road. (model.py - HUG_CORRECTION). Note that for the left & right camera images, this correction comes on top of the already applied steering angle correction.

I again re-trained the model, and now it successfully navigated both tracks, as shown in the videos at the beginning of this document.


#### 3. Dropout layer to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py - line 353). This was necessary as can be seen from the convergence without and with the dropout layer:

![dropout](https://github.com/ArjaanBuijk/CarND_Behavioral_Cloning_P3/blob/master/images/dropout-influence.gif?raw=true)

#### 4. Model parameter tuning


The model is a regression, using the mean squared error (mse) as the loss function. The minimization of mse is done with an adam optimizer, so the learning rate was not tuned manually (model.py - line 360).

The model uses early stopping, via a keras callback (model.py - line 440).
The ideal number of epochs was 7.

The image pre-processing with CLAHE uses a clip limit of 0.5 (model.py - line 20). The model is not sensitive to the value of this parameter.

The steering angle correction for left & right images is 0.2 (model.py - line 23)

The additional steering angle correction for line-hugging data is 0.2 (model.py - line 38-42)

#### 5. Modification to drive.py

The image data generator applies grayscale and clahe. These are not part of the model saved in model.h5, so it was necessary to update drive.py, to apply the same operations for autonomous mode driving. (drive.py - line 74-75) 

Training & Validation Strategy
---

#### 1. Training & Validation Data

For reasons described above, I created 8 separate training data sets:

| track | direction | location   | # lines |
|-------| --------- | ---------- | ------- |
|    1  |   left    |  center    |   4725  |
|    1  |   right   |  center    |   3924  |
|    1  |   left    |  hug-left  |   5230  |
|    1  |   left    |  hug-right |   6430  |
|    2  |   left    |  center    |   4855  |
|    2  |   right   |  center    |   8179  |
|    2  |   left    |  hug-left  |   7233  |
|    2  |   left    |  hug-right |   6479  |
|       |           |<b>TOTAL</b>|<b>47055</b>|

After reading the lines from the log files, I shuffled them into random order. (model.py - line 59)

Each "line" contains 3 images (center, left, right cameras).

To create additional images, I also flipped each image horizontal.

So, in total, there were <b>282,330</b> images.

I split off 10% for validation, the rest was used for training. (model.py - line 416).

A generator was used to efficiently process the images. It yields 32 images per batch.

For debug purposes, I build an option into the generator to write the augmented images to disk. This is done by setting the argument save_to_dir. This concept was copied from the data generator that is build into Keras and it turned out to be critical to debug the model.

---

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)


#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.
