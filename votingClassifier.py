import cv2
import numpy as np
from keras.models import load_model

# Load the saved model from the h5 file
model1 = load_model('model1.h5')
model2 = load_model('model2.h5')
model3 = load_model('model3.h5')

filename = "no_action/plot16449_cropped.png"
# Define the input image path
path1 = 'plots_1/'+filename
path2 = 'plots_2/'+filename
path3 = 'plots_3/'+filename

def getImageArray(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img_array = np.array(img, dtype='float32') / 255.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

prediction1 = model1.predict(getImageArray(path1))
predicted_class1 = np.argmax(prediction1)

prediction2 = model2.predict(getImageArray(path2))
predicted_class2 = np.argmax(prediction1)

prediction3 = model3.predict(getImageArray(path3))
predicted_class3 = np.argmax(prediction3)

print('Predicted class1:', predicted_class1)
print('Predicted class2:', predicted_class2)
print('Predicted class3:', predicted_class3)
results = [predicted_class1, predicted_class2, predicted_class3]


def max_occuring_value(lst):
    return max(set(lst), key=lst.count)
print("The most predicted class:")
max_occuring_value(results)


max_value = max_occuring_value(results)
print(max_value)
