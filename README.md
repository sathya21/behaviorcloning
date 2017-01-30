#DataSet
  * Used Udacity data for the initial training. Since car was unable to train, augmented it with more images from the turn

#preprocessing
Images from the training were fed into preprocessing steps comprising of
  * Cropping first 22 rows and last 25 rows. First 22 to remove sky and last 25 to remove the hood of the car
  * Then it was resized to 66 * 200 so that it can be fed into nvidia model
  * Apart from center image, left and right were used to train the car, for left images, steering was added with +0.15 and for right image, steering was subtracted with 0.15
  * Removed half of images with steering between -0.01 and 0.001 so that model can give more weights for the turn


#model
   Used nvidia model with the following layers
   *  sequential model for making linear stack of layers
   *  Lambda normalization of values
   *  First Convolution layer of 24*5*5 with 2*2 stride
   *  Second Convolution layer of 36*5*5 with 2*2 stride
   *  Third Convolution layer of 48*5*5 with 2*2 stride
   *  Fourth and Fifth Convolution layer of 64*3*3 with 1*1 stride
   *  Then added 3 Fully connected layers of 100,50 and 10
   *  Learning rate of 0.0001 was used


# Running
   * Cropped the image by 32 rows on top and 25 rows at the bottom . 
   * Dimension of the image was changed to 66 * 200
   * reduced throttle to match the training on turns
