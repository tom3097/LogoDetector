from applications import FcnResNet50
from datasets import H5pyLogos32
from keras.preprocessing import image
from PIL import Image, ImageDraw
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

    window_no = 10
    size = 197 + window_no * WINDOW_STRIDE

    img = image.load_img(image_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)
    encoded_labels = np.argmax(predictions, axis=3)
    labels = labels_encoder.inverse_transform(encoded_labels)

    # true label for the given image
    true_label = ''
    im = Image.fromarray(x.astype('uint8'), 'RGB')
    draw = ImageDraw.Draw(im)

    for i in xrange(labels.shape[1]):
        for j in xrange(labels.shape[2]):
            if labels[0, i, j] == true_label:
                width = 3
                cor = [i * WINDOW_STRIDE, j * WINDOW_STRIDE, i * WINDOW_STRIDE + 196, j * WINDOW_STRIDE + 196]
                draw.text((cor[0] + 5, cor[1] + 5), true_label)
                for w in xrange(width):
                    draw.rectangle(cor, outline='red')
                    cor = [cor[0] + 1, cor[1] + 1, cor[2] + 1, cor[3] + 1]

    im.show()
