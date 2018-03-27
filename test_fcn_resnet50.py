from applications import FcnResNet50
from datasets import H5pyLogos32
from keras.preprocessing import image
from PIL import Image
import numpy as np
from sklearn import preprocessing

WINDOW_STRIDE = 32

weights_path = ''
image_path = ''

if __name__ == '__main__':
    classes_no = len(H5pyLogos32.CLASSES)
    model = FcnResNet50(input_shape=(None, None, 3), classes_no=classes_no, fine_tune=False)

    model.load_weights(weights_path)

    labels_encoder = preprocessing.LabelEncoder()
    labels_encoder.fit(H5pyLogos32.CLASSES)

    window_no = 5
    size = 197 + window_no * WINDOW_STRIDE

    img = image.load_img(image_path, target_size=(size, size))

    x = image.img_to_array(img)

    im = Image.fromarray(x.astype('uint8'), 'RGB')
    im.show()

    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)
    encoded_labels = np.argmax(predictions, axis=3)

    labels = labels_encoder.inverse_transform(encoded_labels)

    print labels
