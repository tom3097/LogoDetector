from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class FullyConvResnet50SL(object):
    def __init__(self, classes_no):
        self.__classes_no = classes_no
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(None,None,3))
        x = base_model.get_layer('activation_49').output
        x = Conv2D(2048, (3, 3), strides=(1, 1))(x)
        predictions = Conv2D(self.__classes_no, (5, 5), activation='softmax')(x)
        self.__model = Model(inputs=base_model.input, outputs=predictions)
        self.sliding_window_stride = 32

    def predict(self, image):
        return self.__model.predict(image)

if __name__ == '__main__':
    fullyConvResnet50SL = FullyConvResnet50SL()

    window_no = 4
    size = 197 + window_no * fullyConvResnet50SL.sliding_window_stride

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(size, size))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = fullyConvResnet50SL.predict(x)

    print np.argmax(predictions, axis=3)