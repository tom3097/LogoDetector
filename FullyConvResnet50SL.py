from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, Dropout
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras import backend as K

from sklearn import preprocessing

from PIL import Image

from keras.layers import Layer

from LogoClassDataGen import LogoClassDataGen

CLASSES = [
    'adidas', 'aldi', 'apple', 'becks', 'bmw', 'carlsberg', 'chimay', 'cocacola',
    'corona', 'dhl', 'erdinger', 'esso', 'fedex', 'ferrari', 'ford', 'fosters',
    'google', 'guiness', 'heineken', 'HP', 'milka', 'nvidia', 'paulaner', 'pepsi',
    'rittersport', 'shell', 'singha', 'starbucks', 'stellaartois', 'texaco',
    'tsingtao', 'ups'
]

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(CLASSES)

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape


class FullyConvResnet50SL(object):
    def __init__(self, classes_no):
        self.__classes_no = classes_no
        self.__base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(None,None,3))
        x = self.__base_model.get_layer('activation_49').output
        x = Dropout(0.5)(x)
        x = Conv2D(self.__classes_no, (3, 3), strides=(1, 1))(x)
        predictions = Conv2D(self.__classes_no, (5, 5), activation='linear')(x)
        predictions = Softmax4D(axis=-1)(predictions)
        self.__model = Model(inputs=self.__base_model.input, outputs=predictions)
        self.sliding_window_stride = 32

    def fine_tune(self):
        for layer in self.__base_model.layers:
            layer.trainable = False

        logoGen = LogoClassDataGen('/home/tomasz/PycharmProjects/LogoDetector/flickrlogos32.h5', 30, 2)
        self.__model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['mse', 'accuracy'])

        self.__model.fit_generator(generator=logoGen.generate('train'), steps_per_epoch=90, epochs=10,
                                   validation_data=logoGen.generate('test'), validation_steps=18)

        self.__model.save('my_model2.h5')

    def predict(self, image):
        return self.__model.predict(image)

    def load_w(self):
        self.__model.load_weights('my_model2.h5')


if __name__ == '__main__':
    fullyConvResnet50SL = FullyConvResnet50SL(33)

    #fullyConvResnet50SL.fine_tune()

    fullyConvResnet50SL.load_w()

    window_no = 10
    size = 197 + window_no * fullyConvResnet50SL.sliding_window_stride
    img_path = 'star.jpg'
    img = image.load_img(img_path, target_size=(size, size))
    x = image.img_to_array(img)
    im = Image.fromarray(x.astype('uint8'), 'RGB')
    im.show()
    f = np.zeros((1, size,size, 3), dtype=int)
    f[0,:, :, :] = np.copy(x)
    predictions = fullyConvResnet50SL.predict(f)
    af = np.argmax(predictions, axis=3)
    print af.shape

    print np.max(predictions, axis=3)

    sd = np.chararray(af.shape, itemsize=20)
    sd[:] = " "
    print "shape"
    print sd.shape
    for i in xrange(predictions.shape[1]):
        for j in xrange(predictions.shape[2]):
            if af[0,i,j] == 32:
                sd[0,i,j] = "BG"
            else:
                sd[0,i,j] = label_encoder.inverse_transform(af[0,i,j])

    #print predictions.shape
    #label_enc_pred = np.argmax(predictions, axis=3)
    #print label_encoder.inverse_transform([label_enc_pred])
    print  sd


    # logoGen = LogoClassDataGen('/home/tomasz/PycharmProjects/LogoDetector/flickrlogos32.h5', 30, 2)
    # x, label = logoGen.get_random_test()
    # f = np.zeros((1, 197,197, 3), dtype=int)
    # f[0,:, :, :] = np.copy(x)
    # im = Image.fromarray(x.astype('uint8'), 'RGB')
    # im.show()
    # predictions = fullyConvResnet50SL.predict(f)
    # print predictions
    # print predictions.shape
    # label_enc_pred = np.argmax(predictions, axis=3)[0][0][0]
    # label_enc = label
    # print label_encoder.inverse_transform([label_enc_pred, label_enc])

    #fullyConvResnet50SL.fine_tune()
