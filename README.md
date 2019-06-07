# Hand_Gesture-Recognition
This project is intended to create an interpreter for recognising hand signs and gestures. As of now it supports 14 hand signs of American Sign Language.

## Motivation
A gesture can be recognized as a form of nonverbal communication in which certain bodily actions represent and communicate particular messages. These actions may be represented by facial features or by hand movement. The hand movements are conveniently deciphered to particular messages that are universal in nature.This project presents a model system that allows the differently abled people to communicate with their counterparts effectively. It focuses on the problem of predicting the appropriate response to a particular hand
sign gesture effectively. The system can be used as a support application to facilitate efficient communication in medical and other emergencies. 

## Description
The hand signs are special gestures when performed establish communication. In emergency situations mute people are unable to provide information as in these tasks it's not possible to carry on  pen-paper. The camera attached to such personnels can facilitate such responses efficiently by identifying the gestures and converting them to speech. Also people who are unable to hear develop a bias towards hand signs and so the implementation of this would suffice them too. 

## Working
#### 1
The user initially has to create up a folder of hand signs and gestures that would act as a dataset for the model. This folder contains the segmented hand signs thresholded and background noise removed. This acts as training and testing set for the neural network model. Launch signCreator.py" and press 'q' to capture and save images.


#### 2
Then these images are converted to a CSV file where there dimension is flattened. This acts the training and testing set. Run "dataCreator.py", to create the CSV file.
####3
Run "modelCreate.py" to initialise the neural network model. It saves the model and when additional data is provided,it continues from the last saved model.

#### 3
Run "main.py" to run the hand gesture recognition. It provides a textual representation of the hand signs performed. Additional audio library can convert the text to speech and speech to text to hand signs conversion.

## Packages Used:
To initialise the project, the following libraries should be included by:
```python
pip install -r requirements.txt
```



