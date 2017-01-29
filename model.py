import csv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import  numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import cv2


def resize(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (66, 200))

def normalize(image):
    '''Normalize the image to be between -0.5 and 0.5'''
    return image / 255.0 - 0.5




def get_model(time_len=1):
  ch, row, col = 3, 66, 200  # camera format

  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
           input_shape=( row, col, ch),
        output_shape=(row, col, ch)))
  #model.add(Cropping2D(cropping=((22, 1), (1, 1)), input_shape=(160,320,3),  dim_ordering="tf"))
  #model.add(Lambda(resize))
  #model.add(Lambda(normalize))
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
  model.add(ELU())


  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
  model.add(ELU())

  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
  model.add(ELU())

  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal'))
  model.add(ELU())

  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal'))
  model.add(ELU())


  model.add(Flatten())
  model.add(Dense(1164, init='he_normal'))
  #model.add(Dropout(.2))
  model.add(ELU())

  model.add(Dense(100, init='he_normal'))
  #model.add(Dropout(.3))

  model.add(ELU())
  model.add(Dense(50, init='he_normal'))
  #model.add(Dropout(.3))

  model.add(ELU())
  model.add(Dense(10, init='he_normal'))
  #model.add(Dropout(.3))

  model.add(ELU())
  model.add(Dense(1, init='he_normal'))

  adam= Adam(lr=0.0001)
  #, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
  model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

  return model

def read_csv():
    with open('./driving_log_new.csv', 'r') as f:
       reader = csv.reader(f)
       driving_list = list(reader)
       j  = 0
       for i in driving_list:

          if ( j !=0 ):
              img=mpimg.imread(i[0])
              crop_img = img[22:320,0:200 ]
              image_resized = cv2.resize(crop_img, (200, 66))
              ##image_resized = cv2.resize(img, (200, 100))
              #crop_img = image_resized[34:100,0:200 ]
              #if(float(i[3]) < -0.01 or float(i[3]) > 0.001):
              X_train_list.append(image_resized)
              y_train_list.append(i[3])

              img1 = mpimg.imread(i[1].strip())
              crop_img1 = img1[22:320,0:200 ]
              image_resized1 = cv2.resize(crop_img1, (200, 66))
              #if(float(i[3]) < -0.01 or float(i[3]) > 0.001):
              X_train_list.append(image_resized1)
              y_train_list.append(float(i[3])+0.15)

              img2 = mpimg.imread(i[2].strip())
              crop_img2 = img2[22:320,0:200]

              image_resized2 = cv2.resize(crop_img2, (200, 66))
              #if(float(i[3]) < -0.01 or float(i[3]) > 0.001):
              X_train_list.append(image_resized2)
              y_train_list.append(float(i[3])-0.15)
              #img=mpimg.imread(i[0])
              #image_resized = cv2.resize(img, (200, 100))
              #crop_img = image_resized[34:100,0:200 ]
              #X_train_list.append(crop_img)
              #y_train_list.append(i[3])
              #img1 = mpimg.imread(i[1].strip())
              #image_resized = cv2.resize(img1, (200, 100))
              #crop_img = image_resized[34:100,0:200 ]
             # X_train_list.append(crop_img)
             # y_train_list.append(float(i[3])+0.15)
              #img1 = mpimg.imread(i[2].strip())
              #image_resized = cv2.resize(img1, (200, 100))
              #crop_img = image_resized[34:100,0:200 ]
              #X_train_list.append(crop_img)
              #y_train_list.append(float(i[3])-0.15)
          j=j+1

X_train_list = []
y_train_list = []
read_csv()
X_train = np.asarray(X_train_list)
y_train = np.asarray(y_train_list)
X_train, y_train = shuffle(X_train, y_train)


X_train, X_validation, y_train, y_validation = train_test_split(
     X_train, y_train, test_size=0.2, random_state=42)

n_train = len(X_train)

n_y_train = len(y_train)

print (n_train)
print (n_y_train)


X_train, y_train = shuffle(X_train, y_train)

model = get_model()
batch_size = 256
nb_epoch = 8

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_validation, y_validation))

model.summary()
json_string = model.to_json()
target = open("./model.json", 'w')
target.write(json_string)
print (json_string)
model.save_weights("./model.h5")
