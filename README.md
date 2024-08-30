# Driver Drowsiness Detection

This project aims to detect the drowsiness of a driver in real-time using video input from a live camera positioned in front of the driver's seat. The system raises an alarm if the driver appears to be drowsy for a certain number of consecutive frames, helping to prevent accidents caused by drowsiness.

## Overview

The system uses a combination of computer vision techniques and deep learning models to analyze the driver's facial features, particularly the eyes, to determine whether they are open or closed. If the eyes remain closed for a predetermined threshold of consecutive frames, an alarm is triggered to alert the driver.

## Libraries Used

```python
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random, shutil
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model
import cv2
from pygame import mixer
import time
```

## Files and Their Functionality

1. **Detection Files:**

   - **face.py**: 
     - Detects faces using the Haar Cascade classifier and draws rectangles around detected faces. 
     - Captures and saves the frame if a face is detected.

   - **eyes.py**:
     - Detects eyes using Haar Cascade classifiers for both left and right eyes.
     - Draws rectangles around detected eyes and displays "Eyes Detected" on the frame.

   - **drowsiness.py**:
     - Combines face and eye detection to monitor the driver's state.
     - Calculates a score based on whether the eyes are closed. If the score exceeds a threshold (20 frames), an alarm is triggered.
     - **Frame Rate Calculation**: The actual frame rate depends on the system's processing capabilities. The frame rate in practice can be lower than 20 FPS, as the following code snippet determines the frame count:


       ```python
       currentframe += 1
       ```
     - The frame rate observed in real-world scenarios may be closer to **10-15 frames per second** depending on the computational load and camera performance.

2. **Model File:**

   - **model.py**:
     - Defines and trains a Convolutional Neural Network (CNN) model to classify whether the driver's eyes are open or closed.
     - The model uses three convolutional layers with dropout and max-pooling to extract features, followed by a fully connected dense layer and an output softmax layer to classify the state of the eyes.
     - Trains the model using images from the `data/train` directory and validates it using images from the `data/valid` directory.
     - Saves the trained model as `cnnCat2.h5`.

## How It Works

1. **Real-Time Detection**:
   - The system captures video input from the camera and processes each frame to detect faces and eyes.
   - If the eyes are detected to be closed for more than 20 consecutive frames, an alarm is sounded to alert the driver.

2. **Threshold Setting**:
   - The system uses a threshold of 20 frames to determine drowsiness. This can be adjusted based on the requirements.

3. **Frame Generation**:
   - The system generates and processes approximately **10-15 frames per second** based on system performance and processing time. The number of frames generated per second is tracked using the `currentframe` variable.

## Conclusion

This project demonstrates a practical application of computer vision and deep learning in enhancing road safety by detecting driver drowsiness. The system effectively identifies signs of drowsiness and alerts the driver, potentially preventing accidents caused by fatigue.



