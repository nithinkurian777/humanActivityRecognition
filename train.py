import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# Set the paths to the main folders containing the subfolders of images
folder1_path = "plots_1"
folder2_path = "plots_2"
folder3_path = "plots_3"

# Set the image dimensions and number of classes
img_height = 75
img_width = 75
num_classes = 4

# Combine the images with the same names from the subfolders
combined_images = []
labels = []

for subfolder in os.listdir(folder3_path):
    if os.path.isdir(os.path.join(folder2_path, subfolder)) and os.path.isdir(os.path.join(folder2_path, subfolder)) and os.path.isdir(os.path.join(folder3_path, subfolder)):
        for filename in os.listdir(os.path.join(folder1_path, subfolder)):
            if filename in os.listdir(os.path.join(folder2_path, subfolder)) and filename in os.listdir(os.path.join(folder3_path, subfolder)):
                img1 = cv2.imread(os.path.join(folder1_path, subfolder, filename), cv2.IMREAD_GRAYSCALE)
                img1 = cv2.resize(img1, (img_height, img_width))
                img2 = cv2.imread(os.path.join(folder2_path, subfolder, filename), cv2.IMREAD_GRAYSCALE)
                img2 = cv2.resize(img2, (img_height, img_width))
                img3 = cv2.imread(os.path.join(folder3_path, subfolder, filename), cv2.IMREAD_GRAYSCALE)
                img3 = cv2.resize(img3, (img_height, img_width))
                combined_img = np.stack([img1, img2, img3], axis=-1)
                combined_images.append(combined_img)
                labels.append(subfolder)

# Convert the combined images and labels to numpy arrays
combined_images = np.array(combined_images)
labels = np.array(labels)

# Convert labels to numerical encoding
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# One-hot encode the labels
labels = to_categorical(labels)

# Shuffle the combined images and labels
indices = np.arange(len(combined_images))
np.random.shuffle(indices)
combined_images = combined_images[indices]
labels = labels[indices]

# Split the combined images and labels into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(combined_images, labels, test_size=0.2)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model to the training data
history = model.fit(train_images, train_labels, epochs=100, batch_size=16, validation_data=(val_images, val_labels))

# Save the model
model.save('model.h5')


# plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# save the plot to disk
plt.savefig('accuracy_1.png')
plt.show()

predictions = model.predict(val_images)
# Convert the predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Print the predicted class labels
print(predicted_labels)

test_loss, test_acc = model.evaluate(val_images, val_labels)

print("Test accuracy:", test_acc)

