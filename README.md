#DataSet
  1) Used Udacity data for the initial training. Since car was unable to train, augmented it with more images from the turn

#preprocessing
Images from the training were fed into preprocessing steps comprising of
  1) Cropping first 22 pixel and last 25 pixel. First 22 to remove sky and last 25 to remove the hood of the car
  2) Then it was resized to 66 * 200 so that it can be fed into nvidia model
  3) Apart from center image, left and right were used to train the car, for left images, steering was added with +0.15 and for right image, steering was subtracted with 0.15
  4) Removed half of images with steering between -0.01 and 0.001 so that model can give more weights for the turn


#model
   Used nvidia model with the following layers
   1)  sequential model for making linear stack of layers
   2)  Lambda normalization of values
   3)  First Convolution layer of 24*5*5 with 2*2 stride
   4)  Second Convolution layer of 36*5*5 with 2*2 stride
   5)  Third Convolution layer of 48*5*5 with 2*2 stride
   6)  Fourth and Fifth Convolution layer of 64*3*3 with 1*1 stride
   7)  Then added 3 Fully connected layers of 100,50 and 10
   8)  Learning rate of 0.0001 was used
