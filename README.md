#DataSet
  * Used Udacity data for the initial training. Since car was unable to train, augmented it with more images from the turn

#preprocessing
Images from the training were fed into preprocessing steps comprising of
  * Cropping first 22 rows and last 25 rows. First 22 to remove sky and last 25 to remove the hood of the car
  * Then it was resized to 66 * 200 so that it can be fed into nvidia model
  * Apart from center image, left and right were used to train the car, for left images, steering was added with +0.15, and for right image steering was subtracted with 0.15
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
   *  Epoch of 8 and batch size of 128 was the best fit based on various iterations

#training
   * Udacity dataset was taken as the base set.
   * But there were number of places where i had to augment the base data as the car was going off the track, hitting the bride or falling into the lake
   * Whereever car goes off  track, i trained the car by recording the turn to bring back the car on track

   ## Training the car not to hit the bridge
   ![center_2017_01_29_21_20_00_386](https://cloud.githubusercontent.com/assets/5102280/22514007/f09a9f0e-e8c3-11e6-930d-e176ec89e77d.jpg)

   ## Training the car not to take turn near the lake
   ![center_2017_01_29_22_30_21_241](https://cloud.githubusercontent.com/assets/5102280/22514052/1680dbb6-e8c4-11e6-92e6-a84163ba5512.jpg)

   ## Traing the car to take turn near the open area
   ![center_2017_01_28_19_32_51_836](https://cloud.githubusercontent.com/assets/5102280/22514867/b4a4726a-e8c6-11e6-9371-3de0787ed10f.jpg)

   ## Data Distribution
   ![histogram](https://cloud.githubusercontent.com/assets/5102280/22516371/9d20ba4a-e8cb-11e6-82e4-4388c7eebb4a.png)
# Running
   * Cropped the image by 32 rows on top and 25 rows at the bottom .
   * Dimension of the image was changed to 66 * 200
   * reduced throttle to match the training on turns
