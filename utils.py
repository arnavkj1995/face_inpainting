from __future__ import division
import math
import os, sys
import json
import random
import h5py
import pprint
import scipy.misc
import numpy as np
import collections
import cv2
import tensorflow as tf
from time import gmtime, strftime

pp = pprint.PrettyPrinter()
F = tf.app.flags.FLAGS

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def get_image(image_path, image_size, is_crop=True, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop)


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True)
    else:
        return scipy.misc.imread(path)


class dataset(object):
    def __init__(self):
        if F.dataset == 'mnist':
            self.data = load_mnist()
        elif F.dataset == 'lsun':
            self.data = lmdb.open(F.data_dir, map_size=500000000000,  # 500 gigabytes
                                  max_readers=100, readonly=True)
        elif F.dataset == 'cifar':
            self.data, self.labels, self.data_unlabelled = load_cifar()
	elif F.dataset == 'celebA':
	    self.data = load_celebA()
        else:
            raise NotImplementedError("Does not support dataset {}".format(F.dataset))

    def batch(self):
        if F.dataset != 'lsun' and F.dataset != 'celebA':
            print "Inside batch size!!!"
            self.num_batches = len(self.data) // F.batch_size
            print "Total number of train", len(self.data)
            print "Total number of test:", len(self.data_unlabelled)
            for i in range(self.num_batches):
                start, end = i * F.batch_size, (i + 1) * F.batch_size
                yield self.data[start:end], self.labels[start:end] , self.data_unlabelled[start:end]

                

        elif F.dataset == 'lsun':
            self.num_batches = 3033042 // F.batch_size
            with self.data.begin(write=False) as txn:
                cursor = txn.cursor()
                examples = 0
                batch = []
                for key, val in cursor:
                    img = np.fromstring(val, dtype=np.uint8)
                    img = cv2.imdecode(img, 1)
                    img = transform(img)
                    batch.append(img)
                    examples += 1

                    if examples >= F.batch_size:
                        batch = np.asarray(batch)[:, :, :, [2, 1, 0]]
                        yield batch
                        batch = []
                        examples = 0
       
        else:
          f = h5py.File("face_annotation.hdf5", "r")
          self.num_batches = len(self.data)// F.batch_size
          print "**************Number of batches will be ::", self.num_batches
          
          for i in range(self.num_batches): 
            sub_list = self.data[i * F.batch_size:(i + 1) * F.batch_size] 
            image_list = []
            landmark_list = []
            for item in sub_list: 
              grp = item
              landmark_list.append(f[grp+'/landmark_matrix'].value)
              image_list.append(get_image(image_path=os.path.join(F.data_dir, item), 
                                            image_size=F.output_size)) 
            sample_images = np.asarray(image_list)
            landmarks = np.asarray(landmark_list)
            yield sample_images, landmarks
          f.close()

def load_celebA():
  file_list = os.listdir(F.data_dir)
  print "Total images found: ", len(file_list)

def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int (h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h: j * h + h, i * w: i * w + w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.toimage(inverse_transform(merge(images, size)), cmax=1.0, cmin=0.0).save(path)


def square_crop(x, npx):
    h, w = x.shape[:2]
    crop_size = min(h, w)
    i = int((h - crop_size) / 2.)
    j = int((w - crop_size) / 2.)
    return scipy.misc.imresize(x[i:i + crop_size, j:j + crop_size], [npx, npx])


def transform(image, npx=128, is_crop=False):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = scipy.misc.imresize(image, [npx, npx]) #square_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.
