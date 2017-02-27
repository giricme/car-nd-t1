# CarND Project 3 - BehavioralCloning

The goal of this project is to train a car to steer around a track. We train a convolutional neural network (CNN) to map images from front-facing cameras directly to steering angles. The system automatically learns internal representations of the necessary processing steps such as detecting useful road features with only the human steering angle as the training signal. We do not explicitly train the model to detect features like lanes, boundaries, dividers etc.

# Data

Training data contains single images paired with the corresponding steering angle in degrees. Udacity provides a simulator to generate this data by using keyboard control to drive the car around the test track. Udacity team also provided us with sample data.

# Architecture

The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers.

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 31, 98, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 14, 47, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       activation_2[0][0]               
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 5, 22, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 3, 20, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       activation_4[0][0]               
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 1, 18, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           activation_5[0][0]               
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        activation_6[0][0]               
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         activation_7[0][0]               
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          activation_8[0][0]               
____________________________________________________________________________________________________
Total params: 252219
____________________________________________________________________________________________________


# Training

One of the interesting aspects of the training data (be it self generated or udacity provided) is that,
it is fairly biased. This is somewhat expected. Most of the time when you drive a car, you are driving 
straight ahead. So most images are associated with steering angle of zero. So we need to figure out how
to remove this inherent bias in the data or work with it as best as we can. 

We split the training data as: 

Training set:  6106  
Validation set:  322  
Test set:  1608

## Data Preprocessing

Images are cropped from 160 x 320 to 76 x 280 so that we focus on the core driving area. Images are resized to (66, 200) as per the NVIDIA paper. Image pixel values are normalized to have values between 0 and 1.

## Data Augmentation

We use a few different techniques of augmentation (a lot of it gathered from Udacity CarND slack channels).

1. Add random brightness to images to simulate driving conditions during various times of the day.

2. Use left and right camera images. These are just right and left shifted images of the car and can simulate the effect of car wandering off to the side, and recovering. We will add a small angle correction to the left camera and subtract a small angle for the right camera. The main idea being the left camera has to move right to get to center, and right camera has to move left. If we assume both left and right cameras are at fixed offset X meters vis a vis the center camera, and if we assume that the car needs to correct by left / right course, back to center, over a distance of Y meters, then trignometrically the correction angle in radians is X / Y (using the small angle approximation for the tan function, viz. tan(a) is a for small values of a). In degress this would be (180/pi) * (X/Y). So for left images, we would correct the provided center_steering as steering = center_steering + (180/pi) * (X/Y) and for right images, we would correct as steering = center_steering - (180/pi) * (X/Y). However after some experimentation we just settled on a value of 0.25 degrees as correction factor.

3. Horizontal shifts. We will shift the camera images horizontally to simulate the effect of car being at different positions on the road, and add an offset corresponding to the shift to the steering angle. We limit the shifts to 10 pixels and use a correction of 0.4 degrees per pixel shift to steering angle.

## Training

### Parameters

1. NVDIA model did not use any drop outs. We tried using drop outs but it did not really help.

2. We used Adam Optimizer with MSE loss function.

3. We trained for 10 epochs with a batch size of 256 samples. MSE tapers off after 6 to 8 epochs.

4. We used Keras generators for generating training data. This prevented us from reading all the 
data into memory.

5. We did not use generators for validation and test data.

6. We trained on MacBook Pro (Retina, 15-inch, Mid 2015) Intel Core i7 Processor 2.2 GHz 16 GB Memory.

### Method

Every run of the generator generates 256 samples. The generator endlessly loops over the training data. For each image from the training set:

1) We randomly choose between center, left and right images.

2) If left or right image, we adjust the steering angle.

3) We read the image from disk into memory.

4) Image is cropped from 160 x 320 to 76 x 280 so that we focus on the core driving area. Image is resized to (66, 200) as per the NVIDIA paper. 

5) Image is randomly adjusted for brightness.

6) Image is flipped randomly with a probablity of 0.5

7) Image is sheared horizontally.

## Testing

I tested the model on test track (no images from this track was used in training). You can see the results at: https://www.youtube.com/watch?v=h-WlD_XvC3c

## Miscellaneous notes

1. Given my keyboard driving skills, even though i did generate training data, i did not end up using that data since it was too noisy. I ended up using Udacity data.

2. I also tried a simpler model with fewer convolution layers and larger fully connected layers. This model had more parameters but given that we had only 6000 images in the training set, this model did a huge under
fit.
