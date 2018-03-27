# -*- coding: utf-8 -*-

import h5py
import numpy as np
from PIL import Image
import io
import random
from LogoClassDataGen import LogoClassDataGen


class RandomTestLogoGen(object):
    """
    RandomTestLogoGen provides random access to test logotypes.
    """
    def __init__(self, h5py_file, width=197, height=197):
        self.__h5py_file = h5py_file
        self.__base = h5py.File(self.__h5py_file, 'r')
        self.__width = width
        self.__height = height

    def get_random_logo_crop(self, random_seed=None):
        """
        Gets random logotype crop and corresponding label from test data.

        :param random_seed: The random seed.
        :return: Crop of logo image and corresponding label.
        """
        np.random.seed(None)

        rand_idx = np.random.randint(0, len(self.__base['test']['images']))
        image = Image.open(io.BytesIO(self.__base['test']['images'][rand_idx]))
        label = self.__base['test']['labels'][rand_idx][0]
        b = self.__base['test']['bboxes'][rand_idx]
        all_bboxes = [(b[i], b[i + 1], b[i + 2], b[i + 3]) for i in range(0, len(b), 4)]
        logo_bbox = random.choice(all_bboxes)

        logo_bbox_gen = LogoClassDataGen.get_logo_crop_box(logo_bbox, image.size)
        logo_crop = image.crop(logo_bbox_gen).resize((self.__width, self.__height), Image.LANCZOS)
        return np.asarray(logo_crop), label
