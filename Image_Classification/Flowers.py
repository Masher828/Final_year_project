from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2


def flower_recog(flower_path):
    path = "Files/Flower_Recognition/"
    generator = image.ImageDataGenerator(horizontal_flip=True, shear_range=0.2, rotation_range=10)
    gen = generator.flow_from_directory(path + "Flowers")
    num_to_flower = {}
    for i in gen.class_indices:
        num_to_flower[gen.class_indices[i]] = i
    # print(num_to_flower)
    model = VGG16(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    # for layer in model.layers:
    #     layer.trainable=False
    flat = Flatten()(model.output)
    d1 = Dense(1000, activation="tanh")(flat)
    d2 = Dense(500, activation="tanh")(d1)
    d3 = Dense(200, activation="tanh")(d2)
    out = Dense(5, activation="softmax")(d3)
    vgg = Model(input=[model.input], output=[out])
    vgg.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    vgg.load_weights(path + "Vgg_model.hdf5")
    img = cv2.imread(flower_path)
    img1 = cv2.resize(img, (256, 256), 3)
    img1 = img1.reshape(1, img1.shape[0], img1.shape[1], img1.shape[2])
    # print(num_to_flower[np.argmax(vgg.predict(img1))])
    # img=cv2.imread(path)
    # cv2.imshow(num_to_flower[np.argmax(model.predict(img1))],img)
    # while (cv2.waitKey(1) != ord("q")):
    #     cv2.imshow(num_to_flower[np.argmax(model.predict(img1))],img)
    return num_to_flower[np.argmax(vgg.predict(img1))]
