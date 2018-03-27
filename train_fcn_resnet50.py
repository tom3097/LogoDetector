from generators import LogoClassDataGen
from applications import FcnResNet50
from datasets import H5pyLogos32
from functools import partial

h5py_path = ''

if __name__ == '__main__':
    data_generator = LogoClassDataGen(h5py_path, logo_per_batch=27,
                                      background_per_batch=5)

    train_generator = partial(data_generator.generate, 'train')
    test_generator = partial(data_generator.generate, 'test')

    classes_no = len(H5pyLogos32.CLASSES)
    model = FcnResNet50(input_shape=(197, 197, 3),classes_no=classes_no,
                        fine_tune=True)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['mse', 'accuracy'])

    # len(train_logo_images) = 1792, len(test_logo_images) = 448
    model.fit_generator(generator=train_generator(), steps_per_epoch=90, epochs=10,
                               validation_data=test_generator(), validation_steps=20)

    model.save('flogos32model.h5')
