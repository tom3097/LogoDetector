import numpy as np
import h5py
from PIL import Image, ImageDraw
import io
import tensorflow as tf

class LogoClassDataGen(object):
    def __init__(self, batch_size, h5py_path):
        self.__batch_size = batch_size
        self.__h5py_path = h5py_path
        self.base = h5py.File(self.__h5py_path, 'r')
        #np.random.seed(4)
        self.dim_x = 197
        self.dim_y = 197
        self.dim_z = 3


    @staticmethod
    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        print("xA %s yA %s xB %s yB %s/n" % (xA, yA, xB, yB))

        #if xA > xB or yA > yB:
        #    return 0.0

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)
        print interArea

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou


    def __get_exploration_order(self, N):
        return np.random.permutation(range(N))

    def __get_coordinates(self, true_box, width, height):
        x_begin = int(true_box[0] - 0.5 * true_box[2])
        x_end = int(true_box[0] + 1.5 * true_box[2])
        y_begin = int(true_box[1] - 0.5 * true_box[3])
        y_end = int(true_box[1] + 1.5 * true_box[3])

        if x_begin < 0:
            print "xbegin"
            x_begin = 0
        if x_end >= width:
            print "xend"
            x_end = width - 1
        if y_begin < 0:
            print "ybegin"
            y_begin = 0
        if y_end >= height:
            print "yend"
            y_end = height - 1

        print "dfdfd"
        print x_begin, x_end, y_begin, y_end

        boxA = np.array([true_box[0], true_box[1], true_box[0]+true_box[2], true_box[1]+true_box[3]])

        boxB = None

        while True:
            xs = np.random.randint(x_begin, x_end, size=2)
            ys = np.random.randint(y_begin, y_end, size=2)

            # sprawdz czy jest ponad ograniczenia
            #
            p1 = [np.min(xs), np.min(ys)]
            p2 = [np.max(xs), np.max(ys)]

            boxB = np.array([p1[0], p1[1], p2[0], p2[1]])
            print boxA
            print boxB

            iou = self.bb_intersection_over_union(boxA, boxB)
            print iou
            if iou >= 0.5:
                break
            #break

        print boxB

        return boxB

    #@staticmethod
    #def sparsify(y):
        'Returns labels in binary NumPy array'
    #    n_classes = 32
    #    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
    #                     for i in range(y.shape[0])])

    def __data_generation(self, id_list):

        print "dfdfdf"
        X = np.empty((self.__batch_size, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.__batch_size), dtype=int)

        # Generate data
        for i, idx in enumerate(id_list):
            img = Image.open(io.BytesIO(self.base['train']['images'][idx]))
            #img.show()

            np_img = np.asarray(img)
            print np_img

            # fixme - ZLE ZAPISANE LABELKI
            label = self.base['train']['labels'][0][idx]
            print "Label"
            print label

            bbox = self.base['train']['bboxes'][idx]
            print "Bboxes"
            print bbox

            d = ImageDraw.Draw(img)
            # fixme moze ? czy nie lepiej od razu zapisac wspolrzedne x1,y1,x2,y2?
            #d.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            #img.show()

            # najpierw crop, potem resize
            width, height = img.size
            print img.size
            coord = self.__get_coordinates(bbox, width, height)
            print coord

            #d.rectangle([coord[0], coord[1], coord[2], coord[3]], outline='red')
            #img.show()

            img2 = img.crop(coord)
            img2 = img2.resize((self.dim_x, self.dim_y), Image.LANCZOS)
            img2.show()

            np_img = np.asarray(img2)

            X[i, :, :, :] = np_img
            y[i] = label

        print y
        print np.array([[1 if y[i] == j else 0 for j in range(32)] for i in range(y.shape[0])])
        # sparsify sie przyda

        return X, y

    def generate(self, group_name):
        print group_name

        print self.base.keys()
        print self.base['train']['images'].shape[0]

        samples_n = self.base[group_name]['images'].shape[0]
        max_iter = int(samples_n / self.__batch_size)


        while True:
            indexes = self.__get_exploration_order(samples_n)

            for i in xrange(max_iter):

                id_list = indexes[i * self.__batch_size: (i+1) * self.__batch_size]
                print id_list

                #Generate data
                X, y = self.__data_generation(id_list)

                print "dfdfdf"
                print X, y

                # pozniej zastapic return z yield
                return X, y

            break

            # indexes to losowa permutacja wszystkich indeksow, jednoznacznie
            # okresla kolejnosc przetwarzania danych


if __name__ == '__main__':
    print "sdfdfdf"
    bc = LogoClassDataGen(4, '/home/tomasz/PycharmProjects/LogoDetector/flickrlogos32.h5')
    print bc
    bc.generate('train')
    print "Dfdf"