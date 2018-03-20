import numpy as np
import h5py
from PIL import Image
import io

# fixme: Image can contain several bounding boxes. I thought that it contains exactly one.


class LogoClassDataGen(object):
    """
    LogoClassDataGen is a data generator used for feeding neural network
    with logotype images and background images. Image is considered to be a logo
    if intersection over union is greater than or equal to 0.5. Image is
    considered to be a background if intersection is equal to 0.0.

    """
    def __init__(self, h5py_file, logo_per_batch, background_per_batch,
                 width=197, height=197, random_seed=65):
        self.__h5py_file = h5py_file
        self.__base = h5py.File(self.__h5py_file, 'r')
        self.__logo_per_batch = logo_per_batch
        self.__background_per_batch = background_per_batch
        self.__batch_size = self.__logo_per_batch + self.__background_per_batch
        self.__width = width
        self.__height = height
        self.__channels = 3
        self.__classes_no = len(self.__base.attrs['classes'].split(', ')) + 1
        self.__background_label = self.__classes_no - 1
        np.random.seed(random_seed)

    @staticmethod
    def __calculate_intersection(bbox_true, bbox_gen):
        """
        Calculates the intersection of two bounding boxes.

        :param bbox_true: First bounding box, refers to the ground truth.
        :param bbox_gen: Second bounding box, refers to the box randomly generated.
        :return: The intersection of two bounding boxes.
        """
        xA = max(bbox_true[0], bbox_gen[0])
        yA = max(bbox_true[1], bbox_gen[1])
        xB = min(bbox_true[2], bbox_gen[2])
        yB = min(bbox_true[3], bbox_gen[3])

        if xA > xB or yA > yB:
            return 0.0

        inter_area = (xB - xA + 1) * (yB - yA + 1)
        return inter_area

    @staticmethod
    def __calculate_iou(bbox_true, bbox_gen):
        """
        Calculates the intersection over union of two bounding boxes.

        :param bbox_true: First bounding box, refers to the ground truth.
        :param bbox_gen: Second bounding box, refers to the one randomly generated.
        :return: The intersection over union of two bounding boxes.
        """
        inter_area = LogoClassDataGen.__calculate_intersection(bbox_true, bbox_gen)
        bbox_true_area = (bbox_true[2] - bbox_true[0] + 1) * (bbox_true[3] - bbox_true[1] + 1)
        bbox_gen_area = (bbox_gen[2] - bbox_gen[0] + 1) * (bbox_gen[3] - bbox_gen[1] + 1)
        iou = inter_area / float(bbox_true_area + bbox_gen_area - inter_area)
        return iou

    @staticmethod
    def __get_exploration_order(data_no):
        """
        Generates order of the exploration.

        :param data_no: The number of data to be analyzed.
        :return: The order of data exploration.
        """
        return np.random.permutation(range(data_no))

    @staticmethod
    def __get_logo_crop_box(bbox, image_size):
        """
        Generates logo bounding box, with intersection over union greater
        or equal to 0.5.

        :param bbox: The ground truth bounding box.
        :param image_size: Size of the image.
        :return: Generated bounding box.
        """
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

    @staticmethod
    def __get_back_crop_box(bbox, image_size):
        """
        Generates background bounding box, with intersection equals to 0.0.

        :param bbox: The ground truth bounding box.
        :param image_size: Size of the image.
        :return: Generated bounding box.
        """
        bbox_true = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        correction = 0
        while True:
            back_width = np.random.randint(min(bbox[2], bbox[3]) - correction, max(bbox[2], bbox[3]) - correction)
            back_height = np.random.randint(min(bbox[2], bbox[3]) - correction, max(bbox[2], bbox[3]) - correction)

            x_rand = np.random.randint(0, image_size[0] - 1 - back_width)
            y_rand = np.random.randint(0, image_size[1] - 1 - back_height)
            bbox_gen = np.array([x_rand, y_rand, x_rand + back_width, y_rand + back_height])
            intersection = LogoClassDataGen.__calculate_intersection(bbox_true, bbox_gen)
            if intersection == 0.0:
                break
            correction += 1

        return bbox_gen

    def __sparsify(self, labels):
        """
        Transforms labels to a binary form (for instance, in a 6-label problem,
        the third label is written [0 0 1 0 0 0]). This form is the only one
        acceptable by the keras.

        :param labels: Labels with classes information.
        :return: Labels transformed to a binary form.
        """
        return np.array([[1 if labels[i] == j else 0 for j in range(self.__classes_no)]
                         for i in range(labels.shape[0])])

    def __data_generation(self, group, indexes_batch):
        """
        Generates a batch of data.

        :param group: 'train' or 'test', refers to train set or test set respectively.
        :param indexes_batch: Indexes of data stored in h5py database.
        :return: Generated batch of data.
        """
        x = np.empty((self.__batch_size, self.__width, self.__height, self.__channels))
        y = np.empty(self.__batch_size, dtype=int)

        background_indexes = np.random.choice(indexes_batch, size=self.__background_per_batch)

        idx = 0
        for db_idx in indexes_batch:
            image = Image.open(io.BytesIO(self.__base[group]['images'][db_idx]))
            label = self.__base[group]['labels'][db_idx][0]
            bbox = self.__base[group]['bboxes'][db_idx]

            logo_bbox_gen = self.__get_logo_crop_box(bbox, image.size)
            logo_crop = image.crop(logo_bbox_gen).resize((self.__width, self.__height), Image.LANCZOS)

            x[idx, :, :, :] = np.asarray(logo_crop)
            y[idx] = label
            idx += 1

            back_no = np.count_nonzero(background_indexes == db_idx)
            for b_idx in xrange(back_no):
                back_bbox_gen = self.__get_back_crop_box(bbox, image.size)
                back_crop = image.crop(back_bbox_gen).resize((self.__width, self.__height),
                                                             Image.LANCZOS)

                x[idx, :, :, :] = np.asarray(back_crop)
                y[idx] = self.__background_label
                idx += 1

        return x, self.__sparsify(y)

    def generate(self, group):
        """
        Feeds a neural network with data, never ending loop.

        :param group: 'train' or 'test', refers to train set or test set respectively.
        :return: None
        """
        data_no = self.__base[group]['images'].shape[0]
        max_iter = int(data_no / self.__logo_per_batch)

        while True:
            indexes = LogoClassDataGen.__get_exploration_order(data_no)
            for i in xrange(max_iter):
                indexes_batch = indexes[i * self.__logo_per_batch: (i + 1) * self.__logo_per_batch]
                x, y = self.__data_generation(group, indexes_batch)
                yield x,y


if __name__ == '__main__':
    h5py_file = ''

    logoClassDataGen = LogoClassDataGen(h5py_file, 16, 16)
    logoClassDataGen.generate('train')
