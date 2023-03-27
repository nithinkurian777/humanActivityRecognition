# Import required libraries
import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

# Define data directory
data_dir = 'plots_1'

# Get list of subdirectories
sub_dirs = [sub_dir for sub_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, sub_dir))]

# Set hyperparameters
batch_size = 32
epochs = 100
input_shape = (128, 128, 3)

# Create train/test split
train_data = []
test_data = []
train_labels = []
test_labels = []

for i, sub_dir in enumerate(sub_dirs):
    path = os.path.join(data_dir, sub_dir)
    images = os.listdir(path)
    train_images, test_images = train_test_split(images, test_size=0.2)
    for img in train_images:
        img_path = os.path.join(path, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, input_shape[:2])
        train_data.append(img_arr)
        train_labels.append(i)
    for img in test_images:
        img_path = os.path.join(path, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, input_shape[:2])
        test_data.append(img_arr)
        test_labels.append(i)

# Convert data and labels to numpy arrays
train_data = np.array(train_data)
test_data = np.array(test_data)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Normalize pixel values to range [0, 1]
train_data = train_data.astype('float32') / 255
test_data = test_data.astype('float32') / 255

# Define data augmentation parameters
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True)

# Build model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(len(sub_dirs), activation='softmax'))

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit model
model.fit(datagen.flow(train_data, train_labels, batch_size=batch_size),
          epochs=epochs,
          validation_data=(test_data, test_labels))

# Evaluate model on test set
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
model.save('model1.h5')