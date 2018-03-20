import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
import numpy as np
import h5py

# fixme: Image can contain several bounding boxes. I thought that it contains exactly one.


class H5pyLogos32(object):
    """
    H5pyLogos32 is a tool used for creating h5py format database for
    FlickrLogos-32 dataset (http://www.multimedia-computing.de/flickrlogos/).

    """
    CLASSES = [
        'adidas', 'aldi', 'apple', 'becks', 'bmw', 'carlsberg', 'chimay', 'cocacola',
        'corona', 'dhl', 'erdinger', 'esso', 'fedex', 'ferrari', 'ford', 'fosters',
        'google', 'guiness', 'heineken', 'HP', 'milka', 'nvidia', 'paulaner', 'pepsi',
        'rittersport', 'shell', 'singha', 'starbucks', 'stellaartois', 'texaco',
        'tsingtao', 'ups'
    ]

    def __init__(self):
        self.__root_directory = None
        self.__train_size = None
        self.__random_state = None
        self.__random_seed = None
        self.__label_encoder = preprocessing.LabelEncoder()
        self.__label_encoder.fit(H5pyLogos32.CLASSES)

    @property
    def __data_path(self):
        """
        Gets the path to the file that contains information about
        image names and labels. The path is created based on the
        root directory and the structure of FlickrLogos-32 directory.

        :return: The path to the file that contains information about
        images and labels.
        """
        return os.path.join(self.__root_directory, 'all.spaces.txt')

    @property
    def __images_path(self):
        """
        Gets the path to the directory that contains images. The path
        is created based on the root directory and the structure of
        FlickrLogos-32 directory.

        :return: The path to the directory that contains images.
        """
        return os.path.join(self.__root_directory, 'flat', 'jpg')

    @property
    def __bboxes_path(self):
        """
        Gets the path to the directory that contains bounding boxes.
        The path is created based on the root directory and the
        structure of the FlickrLogos-32 directory.

        :return: The path to the directory that contains bounding boxes.
        """
        return os.path.join(self.__root_directory, 'flat', 'masks')

    def __prepare_train_test(self):
        """
        Prepares sets for training and testing. For each class, split ratio
        (train, test) = (0.8, 0.2).

        :return: Tuple that consists of train set and test set.
        """
        with open(self.__data_path, 'r') as f:
            train_test = f.read().splitlines()
        train_test = [p.split() for p in train_test]

        class_bucket = {}
        for class_name, file_name in train_test:
            if class_name not in class_bucket:
                class_bucket[class_name] = []
            class_bucket[class_name].append((class_name, file_name))
        # images without logotypes are redundant
        class_bucket.pop('no-logo')

        train_set = []
        test_set = []
        for bucket in class_bucket.values():
            train_part, test_part = train_test_split(bucket, train_size=self.__train_size,
                                                     random_state=self.__random_state)
            train_set.extend(train_part)
            test_set.extend(test_part)
        random.seed(self.__random_seed)
        random.shuffle(train_set)
        random.shuffle(test_set)

        return train_set, test_set

    def __fill_in_h5py_group(self, images, labels, bboxes, set):
        """
        Fills h5py group with appropriate data (images, labels, bounding
        boxes).

        :param images: H5py dataset with images to be filled in.
        :param labels: H5py dataset with labels to be filled in.
        :param bboxes: H5py dataset with bounding boxes to be filled in.
        :param set: List of (class name, file name) tuples, source of
        information.
        :return: None
        """
        for idx, pair in enumerate(set):
            _, file_name = pair
            with open(os.path.join(self.__images_path, file_name), 'rb') as f:
                binary_data = f.read()
            images[idx] = np.fromstring(binary_data, dtype='uint8')

        class_names = [p[0] for p in set]
        encoded_names = self.__label_encoder.transform(class_names)
        labels[:] = [[class_code] for class_code in encoded_names]

        for idx, pair in enumerate(set):
            _, file_name = pair
            with open(os.path.join(self.__bboxes_path, file_name + '.bboxes.txt')) as f:
                rect = f.read().splitlines()
            box = [np.int(p) for p in rect[1].split()]
            bboxes[idx] = box

    def __create_h5py_base(self, train_set, test_set):
        """
        Creates h5py database with groups: 'train', 'test'. Each group consists
        of sets: 'images', 'labels', 'bboxes'. They refer to images, class names
        and bounding boxes, respectively.

        :param train_set: Train set.
        :param test_set: Test set.
        :return: None
        """
        h5py_file = h5py.File('flickrlogos32.h5', 'w')
        uint8_dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        int_dt = h5py.special_dtype(vlen=np.dtype('int'))

        # store class list for reference class ids
        h5py_file.attrs['classes'] = np.string_(', '.join(H5pyLogos32.CLASSES))

        train_group = h5py_file.create_group('train')
        test_group = h5py_file.create_group('test')

        train_images = train_group.create_dataset(
            'images', shape=(len(train_set), ), dtype=uint8_dt
        )
        train_labels = train_group.create_dataset(
            'labels', shape=(len(train_set), ), dtype=int_dt
        )
        train_bboxes = train_group.create_dataset(
            'bboxes', shape=(len(train_set), ), dtype=int_dt
        )

        test_images = test_group.create_dataset(
            'images', shape=(len(test_set), ), dtype=uint8_dt
        )
        test_labels = test_group.create_dataset(
            'labels', shape=(len(test_set), ), dtype=int_dt
        )
        test_bboxes = test_group.create_dataset(
            'bboxes', shape=(len(test_set), ), dtype=int_dt
        )

        self.__fill_in_h5py_group(train_images, train_labels, train_bboxes, train_set)
        self.__fill_in_h5py_group(test_images, test_labels, test_bboxes, test_set)

        h5py_file.close()

    def __initialize(self, root_directory, train_size, random_state, random_seed):
        """
        Initializes state of the object.

        :param root_directory: The path to the 'FlickrLogos-v2' directory (downloaded
        from http://www.multimedia-computing.de/flickrlogos/).
        :param train_size: The ratio of data used for training.
        :param random_state: The random state used as a seed for 'sklearn' module
        methods.
        :param random_seed: The random seed used as a seed for 'random' module methods.
        :return: None
        """
        self.__root_directory = root_directory
        self.__train_size = train_size
        self.__random_state = random_state
        self.__random_seed = random_seed

    def __deinitialize(self):
        """
        Deinitializes state of the object.

        :return: None
        """
        self.__root_directory = None
        self.__train_size = None
        self.__random_state = None
        self.__random_seed = None

    def __call__(self, root_directory, train_size=0.8, random_state=4, random_seed=126):
        """
        Launches the creation of h5py database.

        :param root_directory: The path to the 'FlickrLogos-v2' directory (downloaded
        from http://www.multimedia-computing.de/flickrlogos/).
        :param train_size: The ratio of data used for training.
        :param random_state: The random state used as a seed for 'sklearn' module
        methods.
        :param random_seed: The random seed used as a seed for 'random' module methods.
        :return: None
        """
        self.__initialize(root_directory, train_size, random_state, random_seed)
        train_set, test_set = self.__prepare_train_test()
        self.__create_h5py_base(train_set, test_set)
        self.__deinitialize()


if __name__ == '__main__':
    root_directory = ''

    h5pyLogos32 = H5pyLogos32()
    h5pyLogos32(root_directory)
