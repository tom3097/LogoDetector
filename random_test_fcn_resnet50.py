from applications import FcnResNet50
from datasets import H5pyLogos32
from generators import RandomTestLogoGen
from sklearn import preprocessing
from PIL import Image
import numpy as np


weights_path = ''
h5py_path = ''

if __name__ == '__main__':
    classes_no = len(H5pyLogos32.CLASSES)
    model = FcnResNet50(input_shape=(None, None, 3), classes_no=classes_no, fine_tune=False)

    model.load_weights(weights_path)

    labels_encoder = preprocessing.LabelEncoder()
    labels_encoder.fit(H5pyLogos32.CLASSES)

    generator = RandomTestLogoGen(h5py_file=h5py_path)

    image, label = generator.get_random_logo_crop()

    im = Image.fromarray(image.astype('uint8'), 'RGB')
    im.show()

    x = np.expand_dims(image, axis=0)
    predictions = model.predict(x)
    encoded_label = np.argmax(predictions, axis=3)[0,0,0]
    labels = labels_encoder.inverse_transform([encoded_label, label])

    print("---------------------------------------")
    print("True label: %s" % labels[1])
    print("Predicted label: %s" % labels[0])
    print("---------------------------------------")
