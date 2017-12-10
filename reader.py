from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import scipy.misc

import tensorflow as tf

def read_and_decode(filename_queue, batch_size):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'landmark_height': tf.FixedLenFeature([], tf.int64),
        'landmark_width': tf.FixedLenFeature([], tf.int64),
        'landmark_depth': tf.FixedLenFeature([], tf.int64),
        'landmark_raw': tf.FixedLenFeature([], tf.string),
        'image_raw': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    landmarks = tf.decode_raw(features['landmark_raw'], tf.uint8)
   
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    
    landmark_height = tf.cast(features['landmark_height'], tf.int32)
    landmark_width = tf.cast(features['landmark_width'], tf.int32)
    landmark_depth = tf.cast(features['landmark_depth'], tf.int32)

    image_shape = tf.stack([height, width, depth])
    landmark_shape = tf.stack([landmark_height, landmark_width, landmark_depth])
    
    image = tf.reshape(tf.cast(image, tf.float32), image_shape)
    landmarks = tf.reshape(tf.cast(landmarks, tf.float32), landmark_shape)
    
    image.set_shape((128, 128, 3))
    landmarks.set_shape((128, 128, 3))
    
    images, annotations = tf.train.shuffle_batch([image, landmarks],
                                                 batch_size=batch_size,
                                                 num_threads=16,
                                                 capacity=10000,
                                                 min_after_dequeue=1000)
    
    return images, annotations

if __name__ =='__main__':
    tfrecords_filename = [x for x in os.listdir('test_records/')]
    filename_queue = tf.train.string_input_producer(
                            tfrecords_filename, num_epochs=None)

    images, keypoints = read_and_decode(filename_queue, 64)

    with tf.Session() as sess:
      # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(2048):
            # Retrieve a single instance:
            example, label = sess.run([images, keypoints])
            scipy.misc.imsave('samples_complete/img' + str(i) + '.png', example[0])
            scipy.misc.imsave('samples_complete/ky' + str(i) + '.png', label[0])

        coord.request_stop()
        coord.join(threads)