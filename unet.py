# import json
# with open('~/.keras/keras.json', 'w') as f:
#     f.write(json.dumps({
#          "epsilon": 1e-07,
#          "backend": "tensorflow",
#          "floatx": "float32",
#          "image_data_format": "channels_last"
#     }

import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import dataProcess

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image


COLORS = [
    (0, 0, 0),
    (255,255,255),
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (255,255,0),
    (0,255,255),
    (255,0,255),
    (192,192,192),
    (128,128,128),
    (128,0,0),
    (128,128,0),
    (0,128,0),
    (128,0,128),
    (0,128,128),
    (0,0,128),
]

NP_COLORS = np.array(COLORS, np.uint8)

get_color_r = lambda x: NP_COLORS[x,0]
vec_get_color_r = np.vectorize(get_color_r)
get_color_g = lambda x: NP_COLORS[x,1]
vec_get_color_g = np.vectorize(get_color_g)
get_color_b = lambda x: NP_COLORS[x,2]
vec_get_color_b = np.vectorize(get_color_b)



class Unet(object):
    def __init__(self, img_rows=360, img_cols=480, num_classes=None):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_classes = num_classes
        self.model = None

    def load_data(self, train_dir, test_dir):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_data(train_dir)
        imgs_test = mydata.load_test_data(test_dir)
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print("pool1 shape:", pool1.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print("conv3 shape:", conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        # model = Model(input=inputs, output=conv10)

        conv9 = Conv2D(self.num_classes, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print("conv9 shape:", conv9.shape)

        reshape = Reshape((self.num_classes, self.img_rows * self.img_cols), input_shape=(self.num_classes, self.img_rows, self.img_cols))(conv9)
        print("reshape shape:", reshape.shape)

        permute = Permute((2, 1))(reshape)
        print("permute shape:", permute.shape)

        activation = Activation('softmax')(permute)

        model = Model(input=inputs, output=activation)


        # model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer=Adam(lr=3e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        # print(model.get_config())

        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data('files/train_ds', 'files/test_ds')
        print("loading data done")
        model = self.get_unet()
        print("got unet")

        # model_checkpoint = ModelCheckpoint("unet-weights-improvement-{epoch:02d}-{loss:.2f}.hdf5", monitor='loss', verbose=1, save_best_only=True)
        model_checkpoint = ModelCheckpoint("unet4-{epoch:02d}-{acc:.2f}.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='max')
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=50, verbose=2, shuffle=True,
                  callbacks=[model_checkpoint])

        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('imgs_mask_test.npy', imgs_mask_test)

    @classmethod
    def load(cls, fpath, h=640, w=640, num_classes=7):
        m = cls(h, w, num_classes=num_classes)
        m.model = m.get_unet()
        m.model.load_weights(fpath)
        return m

    def test(self, img_path, out_path=None):
        img_data = img_to_array(load_img(img_path))
        p = self.model.predict(np.array([img_data]), batch_size=1, verbose=0)
        print(p.shape)
        print(p)

        p = np.argmax(p, axis=-1)[0]


        print(p)
        r = np.ndarray((p.shape[0], 3), np.uint8)
        # r = vec_get_color(p)
        r[:,0] = vec_get_color_r(p)
        r[:,1] = vec_get_color_g(p)
        r[:,2] = vec_get_color_b(p)
        # print(r.shape)

        r = r.reshape((self.img_rows, self.img_cols, 3))
        print('r.shape', r.shape)
        print(r)

        if out_path:
            im = Image.fromarray(r)
            im.save(out_path)


if __name__ == '__main__':
    n = Unet(640, 640, 7)
    n.train()

