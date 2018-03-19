import numpy as np
import h5py
from PIL import Image
import io


class LogoClassDataGen(object):
    """
    LogoClassDataGen is a data generator used for feeding a network with:
      - background images
      - logotype crops with intersection over union >= 0.5
    
    """
    def __init__(self, h5py_path, batch_size=32, width=197, height=197, classes_no=32,
                 random_seed=65):
        self.__h5py_path = h5py_path
        self.__base = h5py.File(self.__h5py_path, 'r')
        self.__batch_size = batch_size
        self.__width = width
        self.__height = height
        self.__channels = 3
        self.__classes_no = classes_no
        np.random.seed(random_seed)

    @staticmethod
    def __calculate_iou(bbox_true, bbox_gen):
        xA = max(bbox_true[0], bbox_gen[0])
        yA = max(bbox_true[1], bbox_gen[1])
        xB = min(bbox_true[2], bbox_gen[2])
        yB = min(bbox_true[3], bbox_gen[3])

        if xA > xB or yA > yB:
            return 0.0

        inter_area = (xB - xA + 1) * (yB - yA + 1)
        bbox_true_area = (bbox_true[2] - bbox_true[0] + 1) * (bbox_true[3] - bbox_true[1] + 1)
        bbox_gen_area = (bbox_gen[2] - bbox_gen[0] + 1) * (bbox_gen[3] - bbox_gen[1] + 1)
        iou = inter_area / float(bbox_true_area + bbox_gen_area - inter_area)
        return iou

    @staticmethod
    def __get_exploration_order(data_no):
        return np.random.permutation(range(data_no))

    @staticmethod
    def __get_crop_box(bbox, image_size):
        x_min = bbox[0] - bbox[2]
        x_max = bbox[0] + 2 * bbox[2]
        y_min = bbox[1] - bbox[3]
        y_max = bbox[1] + 2 * bbox[3]

        if x_min < 0:
            x_min = 0
        if x_max >= image_size[0]:
            x_max = image_size[0] - 1
        if y_min < 0:
            y_min = 0
        if y_max >= image_size[1]:
            y_max = image_size[1] - 1

        bbox_true = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        while True:
            x_rand = np.random.randint(x_min, x_max, size=2)
            y_rand = np.random.randint(y_min, y_max, size=2)
            bbox_gen = np.array([np.min(x_rand), np.min(y_rand), np.max(x_rand), np.max(y_rand)])
            iou = LogoClassDataGen.__calculate_iou(bbox_true, bbox_gen)
            if iou >= 0.5:
                break

        return bbox_gen

    def __sparsify(self, labels):
        return np.array([[1 if labels[i] == j else 0 for j in range(self.__classes_no)]
                         for i in range(labels.shape[0])])

    def __data_generation(self, group, indexes_batch):
        x = np.empty((self.__batch_size, self.__width, self.__height, self.__channels))
        y = np.empty(self.__batch_size, dtype=int)

        for idx, db_idx in enumerate(indexes_batch):
            image = Image.open(io.BytesIO(self.__base[group]['images'][db_idx]))
            label = self.__base[group]['labels'][db_idx][0]
            bbox = self.__base[group]['bboxes'][db_idx]
            bbox_gen = self.__get_crop_box(bbox, image.size)
            image = image.crop(bbox_gen).resize((self.__width, self.__height), Image.LANCZOS)

            x[idx, :, :, :] = np.asarray(image)
            y[idx] = label

        return x, self.__sparsify(y)

    def generate(self, group):
        data_no = self.__base[group]['images'].shape[0]
        max_iter = int(data_no / self.__batch_size)

        while True:
            indexes = LogoClassDataGen.__get_exploration_order(data_no)
            for i in xrange(max_iter):
                indexes_batch = indexes[i * self.__batch_size: (i + 1) * self.__batch_size]
                x, y = self.__data_generation(group, indexes_batch)
                return x,y


if __name__ == '__main__':
    logoClassDataGen = LogoClassDataGen('/home/tomasz/PycharmProjects/LogoDetector/flickrlogos32.h5', 32, 197, 197, 32)
    logoClassDataGen.generate('train')