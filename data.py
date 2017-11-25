from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os

from PIL import Image, ImageChops, ImageOps
# import cv2


class dataProcess(object):

    def __init__(self, height, width, img_type="png", num_class=7):
        self.allowed_formats = ['.bmp']

        self.height = height
        self.width = width
        self.img_type = img_type
        self.num_class = num_class

    def binarylab(self, labels):
        x = np.zeros([self.height, self.width, self.num_class])
        for i in range(self.height):
            for j in range(self.width):
                x[i, j, int(labels[i][j])] = 1
        return x

    def create_data_masked(self, img_dir, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)

        data = []
        labels = []

        for f in os.listdir(img_dir):

            name, ext  = os.path.splitext(f)
            if ext not in self.allowed_formats:
                continue
            print(f)
            img_path = os.path.join(img_dir, f)
            img_data = img_to_array(load_img(img_path))
            mask_data = np.zeros((self.height, self.width, self.num_class))
            for label_id in range(1, self.num_class):
                mask_path = os.path.join(os.path.join(img_dir, name, '{}_{}.{}'.format(name, label_id, self.img_type)))
                img_label_data = (img_to_array(load_img(mask_path))[:, :, 0] > 0).astype(int)
                mask_data[:, :, label_id -1] = img_label_data
            data.append(img_data)
            labels.append(mask_data)

        data = np.array(data, dtype=np.uint8)
        labels = np.array(labels, dtype=np.uint8)

        print('loading done')
        labels = np.reshape(labels, (len(labels), self.width * self.height, self.num_class))
        # labels = np.reshape(labels, (len(labels), self.width, self.height, self.num_class))
        np.save(out_dir + '/data.npy', data)
        np.save(out_dir + '/labels.npy', labels)
        print('Saving to .npy files done.')

    def create_data(self, img_dir, mask_dir, outdir):
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)

        data = []
        labels = []

        for f in os.listdir(img_dir):
            img_path = os.path.join(img_dir, f)
            mask_path = os.path.join(mask_dir, f)

            img_data = img_to_array(img_path)
            img_labels = self.binarylab(img_to_array(mask_path)[:, :, 0])

            data.append(img_data)
            labels.append(img_labels)

        data = np.array(data, dtype=np.uint8)
        labels = np.array(labels, dtype=np.uint8)

        print('loading done')
        labels = np.reshape(labels, (len(labels), self.width * self.height, self.num_class))
        np.save(outdir + '/data.npy', data)
        np.save(outdir + '/labels.npy', labels)
        print('Saving to .npy files done.')

    def load_data(self, dir):
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        imgs_train = np.load(dir + "/data.npy")
        imgs_mask_train = np.load(dir + "/labels.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        mean = imgs_train.mean(axis=0)
        imgs_train -= mean
        return imgs_train, imgs_mask_train

    def load_test_data(self, dir):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(dir + "/data.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        mean = imgs_test.mean(axis=0)
        imgs_test -= mean
        return imgs_test


if __name__ == "__main__":
    pass
    # # aug = myAugmentation()
    # # aug.Augmentation()
    # # aug.splitMerge()
    # # aug.splitTransform()
    # mydata = dataProcess(360, 480)
    # mydata.create_train_data()
    # mydata.create_test_data()
    # # imgs_train,imgs_mask_train = mydata.load_train_data()
    # # print imgs_train.shape,imgs_mask_train.shape

    # d.create_data('sample/img', 'sample/img_mask')

    d = dataProcess(640, 640, img_type='bmp')
    d.create_data_masked('files/train', 'files/train_ds')
    d.create_data_masked('files/test', 'files/test_ds')