# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 19:54:34 2020

@author: Connor
@environment: GPU
"""

# Convolutional Neural Network

# Importing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Dropout, Flatten, Dense, MaxPooling2D, BatchNormalization

#Change file path root
file_path = ''

# Initialise the CNN
classifier = Sequential()

# Convolutional Layer
classifier.add(Convolution2D(32, (3, 3), input_shape=(180, 180, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization())

# Add another Convolutional Layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization())

# Add another Convolutional Layer
classifier.add(Convolution2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(BatchNormalization())

# Flattening
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(128, activation = 'relu', kernel_initializer='uniform'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=96,activation='relu',kernel_initializer='uniform'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units=64,activation='relu',kernel_initializer='uniform'))
classifier.add(Dropout(0.2))
classifier.add(Dense(1, activation = 'sigmoid',kernel_initializer='uniform'))

# Compile
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Image Pre-Processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        file_path + 'dataset/training_set',
        target_size=(180, 180),
        batch_size=128,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        file_path + 'dataset/test_set',
        target_size=(180, 180),
        batch_size=128,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/128,
        epochs=100,
        validation_data=test_set,
        validation_steps=2000/128,
        workers = 4
        )

print('Model finished')

## Making a prediction
#import numpy as np
#from tensorflow.keras.preprocessing import image
#test_image = image.load_img(file_path + 'dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict(test_image)
#training_set.class_indeces
#if result[0][0] == 1:
#    prediction = 'Dog'
#else:
#    prediction = 'Cat'
#
#print('The prediction was ' + prediction)


















