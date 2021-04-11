import numpy as np
import cv2
import os


def read_data(data_folder='data', resize_shape=(200, 200)):
    """
    parameters:
        - data_folder = the folder where the data is located
        - resize_shape = resize every image to this shape
    returns:
        - train_images = np.array of RGB images for training
        - train_labels = np.array of labels for each training image
        - test_images = np.array of RGB images for testing
        - test_labels = np.array of labels for each testing image
    """
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for image_folder in os.listdir(data_folder):
        images = []
        labels = []

        print(f'Started reading images from class {image_folder}.')

        for image_name in os.listdir(f'{data_folder}/{image_folder}'):
            images.append(cv2.resize(cv2.cvtColor(cv2.imread(f'{data_folder}/{image_folder}/{image_name}'), cv2.COLOR_BGR2RGB), resize_shape))
            labels.append(image_folder)

        print(f'Finished reading images from class {image_folder}.')

        # 100 images for testing, the rest for training
        train_images += images[:-100]
        train_labels += labels[:-100]
        test_images += images[-100:]
        test_labels += labels[-100:]

    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)


X_train, y_train, X_test, y_test = read_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
