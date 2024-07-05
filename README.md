# Vision-Guard: Diabetic Retinopathy Detection System 

VisionGuard is a deep learning-based system to detect diabetic retinopathy from retinal images. This project aims to provide accurate and efficient detection of diabetic retinopathy, leveraging convolutional neural networks (CNNs) and modern computer vision techniques.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Architecture](#architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Acknowledgement](#acknowledgement)

## Introduction

Diabetic retinopathy is a common complication of diabetes and a leading cause of blindness. VisionGuard utilizes deep learning to automate the detection process, aiding healthcare professionals in early diagnosis and treatment planning.

![image](https://github.com/PixelPirate-Kartikey/Vision-Guard/assets/104156929/aca069d0-1df8-462c-b572-7ff6134f1a72)


## Features

- **Automated Detection**: Detects diabetic retinopathy from retinal images.
- **Model Training**: Trainable on custom datasets for specific applications.
- **Efficient**: Optimized for performance using PyTorch and GPU acceleration.

## Dataset
The dataset used for training VisionGuard should include annotated retinal images. Ensure the dataset is properly preprocessed and split into training, validation, and test sets.

## Architecture
Vision-Guard employs a Convolutional neural network (CNN) architecture designed to extract features from the retinal images and classify them into diabetic retinopathy severity levels.

-----------------------------------------------------------------
        Layer (type)               Output Shape           Param
-----------------------------------------------------------------
            Conv2d-1          [-1, 8, 255, 255]             224
       BatchNorm2d-2          [-1, 8, 255, 255]              16
            Conv2d-3         [-1, 16, 127, 127]           1,168
       BatchNorm2d-4         [-1, 16, 127, 127]              32
            Conv2d-5           [-1, 32, 63, 63]           4,640
       BatchNorm2d-6           [-1, 32, 63, 63]              64
            Conv2d-7           [-1, 64, 31, 31]          18,496
       BatchNorm2d-8           [-1, 64, 31, 31]             128
            Linear-9                  [-1, 100]       1,440,100
           Linear-10                   [-1, 50]           5,050
           Linear-11                    [-1, 2]             102
----------------------------------------------------------------
Total params: 1,470,020  
Trainable params: 1,470,020  
Non-trainable params: 0  
Input size (MB): 0.74  
Forward/Backward pass size (MB): 14.75  
Params size (MB): 5.61  
Estimated Total Size (MB): 21.10  


## Training
Train VisionGuard using the provided dataset and adjust hyperparameters such as learning rate, batch size, and optimizer settings to achieve optimal performance.

## Evaluation
Evaluate VisionGuard using metrics such as accuracy, precision, recall, and F1-score on the test set to assess its performance in detecting diabetic retinopathy.
![image](https://github.com/PixelPirate-Kartikey/Vision-Guard/assets/104156929/31b95346-9227-48e1-8ceb-6e8ae9e81f82)


## Results
### Training Set
                precision   recall  f1-score   support

         0.0       0.97      0.96      0.96      1050
         1.0       0.96      0.97      0.96      1026

    accuracy                           0.96      2076
   macro avg       0.96      0.96      0.96      2076
weighted avg       0.96      0.96      0.96      2076
 


Accuracy: 0.9639

Confusion Matrix:
 [[1007   43]
 [  32  994]]


### Validation Set
                precision  recall   f1-score   support

         0.0       0.97      0.94      0.95       245
         1.0       0.95      0.97      0.96       286

    accuracy                           0.96       531
   macro avg       0.96      0.96      0.96       531
weighted avg       0.96      0.96      0.96       531
 


Accuracy: 0.9586

Confusion Matrix:
 [[231  14]
 [  8 278]]

### Testing Set
                precision    recall  f1-score   support

         0.0       0.97      0.92      0.95       113
         1.0       0.93      0.97      0.95       118

    accuracy                           0.95       231
   macro avg       0.95      0.95      0.95       231
weighted avg       0.95      0.95      0.95       231
 


Accuracy: 0.9481

Confusion Matrix:
 [[104   9]
 [  3 115]]

## Acknowledgement
- Dataset: https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy
- https://www.kaggle.com/code/farheenshaukat/diagnosis-of-diabetic-retinopathy/notebook



