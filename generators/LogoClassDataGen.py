# -*- coding: utf-8 -*-

import numpy as np
import h5py
from PIL import Image
from sklearn import preprocessing
import io
import random


class LogoClassDataGen(object):
    """
    LogoClassDataGen is a data generator used for feeding neural network
    with logotype images and background images. Image is considered to be a logo
    if intersection over union is greater than or equal to 0.5. Image is
    considered to be a background if intersection is equal to 0.0, or if
    it comes from the image without logotype.

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
        self.__classes_no = len(self.__base.attrs['classes'].split(', '))
        self.__label_encoder = preprocessing.LabelEncoder()
        self.__label_encoder.fit(self.__base.attrs['classes'].split(', '))
        self.__background_label = self.__label_encoder.transform(['bg'])[0]
        np.random.seed(random_seed)
        random.seed(random_seed)

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
    def __get_back_crop_box(bboxes, image_size, window_size):
        """
        Generates background bounding box, with intersection equals to 0.0.

        :param bbox: List of ground truth bounding boxes.
        :param image_size: Size of the image.
        :param window_size: Size of the sliding window.
        :return: Generated bounding box, or 'None' if such a box cannot
        be generated.
        """
        if image_size[0] <= window_size[0] + 32 or image_size[1] <= window_size[1] + 32:
            return None

        max_iterations = 50
        iteration = 0

        bbox_found = False
        while not bbox_found:
            if iteration == max_iterations:
                return None

            bbox_found = True

            back_width = np.random.randint(window_size[0] - 32, window_size[0] + 32)
            back_height = np.random.randint(window_size[1] - 32, window_size[1] + 32)

            x_rand = np.random.randint(0, image_size[0] - 1 - back_width)
            y_rand = np.random.randint(0, image_size[1] - 1 - back_height)

            bbox_gen = np.array([x_rand, y_rand, x_rand + back_width, y_rand + back_height])
            for bbox in bboxes:
                bbox_true = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                intersection = LogoClassDataGen.__calculate_intersection(bbox_true, bbox_gen)
                if intersection != 0.0:
                    bbox_found = False
                    break
            iteration += 1

        return bbox_gen

    def __get_background_from_nologo(self, group):
        """
        Gets background from random nologo image.

        :param group: 'train' or 'test', refers to train set or test set respectively.
        :return: Background image.
        """
        rand_idx = np.random.randint(0, len(self.__base[group]['nologos']))
        image = Image.open(io.BytesIO(self.__base[group]['nologos'][rand_idx]))

        back_width = np.random.randint(image.size[0] / 6, image.size[0] / 4)
        back_height = np.random.randint(image.size[1] / 6, image.size[1] / 4)
        x_rand = np.random.randint(0, image.size[0] - 1 - back_width)
        y_rand = np.random.randint(0, image.size[1] - 1 - back_height)

        bbox_gen = np.array([x_rand, y_rand, x_rand + back_width, y_rand + back_height])
        back_crop = image.crop(bbox_gen).resize((self.__width, self.__height),
                                                Image.LANCZOS)
        return back_crop

    def __adjust_labels(self, labels):
        """
        Transforms labels into format appropriate for fully convolutional network.

        :param labels: Labels with classes information.
        :return: Labels adjusted for fully convolutional network.
        """
        fcn_labels = np.zeros((self.__batch_size, 1, 1, self.__classes_no), dtype=int)
        for idx in xrange(self.__batch_size):
            fcn_labels[idx, 0, 0, labels[idx]] = 1
        return fcn_labels

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
            b = self.__base[group]['bboxes'][db_idx]
            all_bboxes = [(b[i], b[i + 1], b[i + 2], b[i + 3]) for i in range(0, len(b), 4)]
            logo_bbox = random.choice(all_bboxes)

            logo_bbox_gen = self.__get_logo_crop_box(logo_bbox, image.size)
            logo_crop = image.crop(logo_bbox_gen).resize((self.__width, self.__height), Image.LANCZOS)

            x[idx, :, :, :] = np.asarray(logo_crop)
            y[idx] = label
            idx += 1

            back_no = np.count_nonzero(background_indexes == db_idx)
            for b_idx in xrange(back_no):
                back_bbox_gen = self.__get_back_crop_box(all_bboxes, image.size,
                                                         (self.__width, self.__height))

                if back_bbox_gen is None:
                    back_crop = self.__get_background_from_nologo(group)
                else:
                    back_crop = image.crop(back_bbox_gen).resize((self.__width, self.__height),
                                                             Image.LANCZOS)

                x[idx, :, :, :] = np.asarray(back_crop)
                y[idx] = self.__background_label
                idx += 1

        return x, self.__adjust_labels(y)

    def get_sample_from_testset(self):
        """
        Gets random logotype image crop and corresponding label from test data.

        :return: Crop of logo image and corresponding label.
        """
        rand_idx = np.random.randint(0, len(self.__base['test']['images']))
        image = Image.open(io.BytesIO(self.__base['test']['images'][rand_idx]))
        label = self.__base['test']['labels'][rand_idx][0]
        b = self.__base['test']['bboxes'][rand_idx]
        all_bboxes = [(b[i], b[i + 1], b[i + 2], b[i + 3]) for i in range(0, len(b), 4)]
        logo_bbox = random.choice(all_bboxes)

        logo_bbox_gen = self.__get_logo_crop_box(logo_bbox, image.size)
        logo_crop = image.crop(logo_bbox_gen).resize((self.__width, self.__height), Image.LANCZOS)
        return np.asarray(logo_crop), label

    def generate(self, group):
        """
        Feeds a neural network with data, never ending loop.

        :param group: 'train' or 'test', refers to train set or test set respectively.
        :return: None.
        """
        data_no = self.__base[group]['images'].shape[0]
        max_iter = int(data_no / self.__logo_per_batch)

        while True:
            indexes = LogoClassDataGen.__get_exploration_order(data_no)
            for i in xrange(max_iter):
                indexes_batch = indexes[i * self.__logo_per_batch: (i + 1) * self.__logo_per_batch]
                x, y = self.__data_generation(group, indexes_batch)
                yield x, y
