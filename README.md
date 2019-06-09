# Hand_Gesture-Recognition
This project is intended to create an interpreter for recognising hand signs and gestures. As of now it supports 14 hand signs of American Sign Language.

## Motivation
A gesture can be recognized as a form of nonverbal communication in which certain bodily actions represent and communicate particular messages. These actions may be represented by facial features or by hand movement. The hand movements are conveniently deciphered to particular messages that are universal in nature.This project presents a model system that allows the differently abled people to communicate with their counterparts effectively. It focuses on the problem of predicting the appropriate response to a particular hand
sign gesture effectively. The system can be used as a support application to facilitate efficient communication in medical and other emergencies. 

## Description
The hand signs are special gestures when performed establish communication. In emergency situations mute people are unable to provide information as in these tasks it's not possible to carry on  pen-paper. The camera attached to such personnels can facilitate such responses efficiently by identifying the gestures and converting them to speech. Also people who are unable to hear develop a bias towards hand signs and so the implementation of this would suffice them too. 

## Working
#### 1
The user initially has to create up a folder of hand signs and gestures that would act as a dataset for the model. This folder contains the segmented hand signs thresholded and background noise removed. This acts as training and testing set for the neural network model. Launch *signCreator.py* script by doing:
```python
python signCreator.py
```
You are then prompted to make a selection. Enter a choice of the hand sign in numerical digit. Press 'q' to capture and save the images.
```python
Enter the hand sign number :: 1
Press q to capture the images
```
<img src="https://github.com/AkhilDixit1998/Hand_Gesture-Recognition/blob/master/screenshots/signs.png" height="450">
Place the hand in the green box and wait for the message to display on the screen for capturing the images. You'll see the camera feed. Move your hand slowly across the frame, closer and further from the camera. Try to rotate a bit your pose. Do every movement slowly as you want to create maximum number of images with minimal changes.

#### 2
Then these images are converted to a CSV file where there dimension is flattened. This acts the training and testing set. Launch dataCreator.py script, to create the CSV file.
```python
python dataCreator.py
```
#### 3
Launch modelCreate.py script to initialise the neural network model. It saves the model and when additional data is provided,it continues from the last saved model.
```python
python modelCreate.py
```
<img src="https://github.com/AkhilDixit1998/Hand_Gesture-Recognition/blob/master/screenshots/modelcreate.gif" height="450">

#### 4
Launch main.py script to run the hand gesture recognition. It provides a textual representation of the hand signs performed. Additional audio library can convert the text to speech and speech to text to hand signs conversion.
```python
python main.py
```
<img src="https://github.com/AkhilDixit1998/Hand_Gesture-Recognition/blob/master/screenshots/recognition.gif" height="450">

## Architecture

### Pipeline
<img src="https://github.com/AkhilDixit1998/Hand_Gesture-Recognition/blob/master/screenshots/pipeline.png"  height="450">
The pipeline of this project consists of 4 steps :

- A frame is grabbed from the camera by a dedicated thread, converted to RGB (from BGR) and put into the input queue

- A worker grabs the frame from the queue and pass it into the SSD. This gives us a bouding box of where the hand(s) is and the corresponding cropped frame.  

- This cropped frame of the hand is then passed to the CNN, which give us a class vector output of values between 0 and 1. These values correspond to the probability of the frame to be one of the classes. The worker has finished its job and put: the frame with bouding box drawn on top, the cropped frame and the classes into three different queues.

- The main thread, responsible of showing the results can grab the informations from the queues and display them in three windows.


## Packages Used:
To initialise the project, the following libraries should be included by:
```python
pip install -r requirements.txt
```

## Versions
At the time of project:
- Python version is is 3.6.1
- Tensorflow version 1.13.1
- Opencv-python is 3.3.1
- Keras is 2.2.4

### Gesture Sheet
The sign sheet used to form the data set is as:
<br>
<img src="https://github.com/AkhilDixit1998/Hand_Gesture-Recognition/blob/master/screenshots/asl.jpg" height="450">




