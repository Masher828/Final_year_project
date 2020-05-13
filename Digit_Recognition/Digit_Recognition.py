import math
import operator


def digit_recognition(path):
    import cv2
    import numpy as np
    # import matplotlib.pyplot as plt
    #
    # # save the final model to file
    # from keras.datasets import mnist
    # from keras.utils import to_categorical
    from keras.models import Sequential
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.optimizers import SGD

    # # load train and test dataset
    # def load_dataset():
    #     print("dataset")
    #     # load dataset
    #     (trainX, trainY), (testX, testY) = mnist.load_data()
    #     # reshape dataset to have a single channel
    #     trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    #     testX = testX.reshape((testX.shape[0], 28, 28, 1))
    #     # one hot encode target values
    #     trainY = to_categorical(trainY)
    #     testY = to_categorical(testY)
    #     return trainX, trainY, testX, testY
    #
    #
    # # scale pixels
    # def prep_pixels(train, test):
    #     print("prep")
    #     # convert from integers to floats
    #     train_norm = train.astype('float32')
    #     test_norm = test.astype('float32')
    #     # normalize to range 0-1
    #     train_norm = train_norm / 255.0
    #     test_norm = test_norm / 255.0
    #     # return normalized images
    #     return train_norm, test_norm
    #
    #
    # # define cnn model
    # def define_model():
    #     model = Sequential()
    #     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    #     model.add(MaxPooling2D((2, 2)))
    #     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    #     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    #     model.add(MaxPooling2D((2, 2)))
    #     model.add(Flatten())
    #     model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    #     model.add(Dense(10, activation='softmax'))
    #     # compile model
    #     opt = SGD(lr=0.01, momentum=0.9)
    #     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #     return model
    #
    #
    # # run the test harness for evaluating a model
    # def run_test_harness():
    #     print("harness")
    #     # load dataset
    #     trainX, trainY, testX, testY = load_dataset()
    #     # prepare pixel data
    #     trainX, testX = prep_pixels(trainX, testX)
    #     # define model
    #     model = define_model()
    #     # fit model
    #     model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
    #     # save model
    #     model.save('final_model.h5')
    #
    #
    # # entry point, run the test harness
    # run_test_harness()
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights("Files/Sudoku_Solver/final_model.h5")

    def pre_process_image(img):
        proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
        proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        img = cv2.bitwise_not(proc)

        return img

    def find_corners_of_largest_polygon(img):
        """Finds the 4 extreme corners of the largest contour in the image."""
        contours, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
        polygon = contours[0]  # Largest image

        # Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
        # Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

        # Bottom-right point has the largest (x + y) value
        # Top-left has point smallest (x + y) value
        # Bottom-left point has smallest (x - y) value
        # Top-right point has largest (x - y) value
        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

        # Return an array of all 4 points using the indices
        # Each point is in its own array of one coordinate
        return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
    def distance_between(p1, p2):
        """Returns the scalar distance between two points"""
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]
        return np.sqrt((a ** 2) + (b ** 2))

    def crop_and_warp(img, crop_rect):
        """Crops and warps a rectangular section from an image into a square of similar size."""

        # Rectangle described by top left, top right, bottom right and bottom left points
        top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

        # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

        # Get the longest side in the rectangle
        side = max([
            distance_between(bottom_right, top_right),
            distance_between(top_left, bottom_left),
            distance_between(bottom_right, bottom_left),
            distance_between(top_left, top_right)
        ])

        # Describe a square with side of the calculated length, this is the new perspective we want to warp to
        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

        # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
        m = cv2.getPerspectiveTransform(src, dst)

        # Performs the transformation on the original image
        return cv2.warpPerspective(img, m, (int(side), int(side)))
    def show_image(img):
        """Shows an image until any key is pressed"""
        cv2.imshow('image', img)  # Display the image
        cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
        cv2.destroyAllWindows()  # Close all window# See: https://gist.github.com/mineshpatel1/22e86200eee86ebe3e221343b26fc3f3#file-show_image-py

    img = cv2.imread(path)
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #processed = pre_process_image(gray)
    corners = find_corners_of_largest_polygon(gray)
    sud = crop_and_warp(gray, corners)
    # if not  path[len(path)-6:len(path)] == "em.jpg":
    #     gray = pre_process_image(gray)

    # corners = find_corners_of_largest_polygon(processed)
    # sud = crop_and_warp(processed, corners)
    # show_image(gray)
    # show_image(sud)
    small = cv2.resize(gray, (28, 28))
    # show_image(small)
    small = np.reshape(small, (1, small.shape[0], small.shape[1], 1))
    pred = model.predict(small)
    ans = np.argmax(pred)
    #show_image(processed)
    print(ans)
    return ans
