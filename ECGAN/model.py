from __future__ import division
import os
import sys
import time
import tensorflow as tf
import numpy as np
import scipy.misc
from six.moves import xrange

import reader
import random
from ops import *
import poissonblending
from utils_errors import *
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr

F = tf.app.flags.FLAGS

class ECGAN(object):
    def __init__(self, sess):
        self.sess = sess
        self.ngf = 128
        self.ndf = 64
        self.nt = 128
        self.k_dim = 16
        self.image_shape = [F.output_size, F.output_size, 3]
        self.build_model()
        if F.output_size == 64:
            self.is_crop = True
        else:
            self.is_crop = False

    def build_model(self):
        # main method for training the conditional GAN

        if F.use_tfrecords == True:
            # load images from tfrecords + queue thread runner for better GPU utilization
            tfrecords_filename = ['train_data/' + x for x in os.listdir('train_data/')]
            filename_queue = tf.train.string_input_producer(
                                tfrecords_filename, num_epochs=100)


            self.images, self.keypoints = reader.read_and_decode(filename_queue, F.batch_size)

            if F.output_size == 64:
                self.images = tf.image.resize_images(self.images, (64, 64))
                self.keypoints = tf.image.resize_images(self.keypoints, (64, 64))

            self.images = (self.images / 127.5) - 1
            self.keypoints = (self.keypoints / 127.5) - 1

        else:    
            self.images = tf.placeholder(tf.float32,
                                       [F.batch_size, F.output_size, F.output_size,
                                        F.c_dim],
                                       name='real_images')
            self.keypoints = tf.placeholder(tf.float32, [F.batch_size, F.output_size, F.output_size, F.c_dim], name='keypts')

        self.is_training = tf.placeholder(tf.bool, name='is_training')        
        self.z_gen = tf.placeholder(tf.float32, [F.batch_size, F.z_dim], name='z')

        self.G = self.generator(self.z_gen, self.keypoints)
        self.D, self.D_logits = self.discriminator(self.images, self.keypoints, reuse=False)
        self.D_, self.D_logits_, = self.discriminator(self.G, self.keypoints, reuse=True)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss_actual = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        if F.error_conceal == True:
            self.mask = tf.placeholder(tf.float32, [F.batch_size] + self.image_shape, name='mask')
            self.contextual_loss = tf.reduce_sum(tf.contrib.layers.flatten(
                                                 tf.abs(tf.multiply(self.mask, self.G) -
                                                 tf.multiply(self.mask, self.images))), 1)
            self.perceptual_loss = self.g_loss_actual
            self.complete_loss = self.contextual_loss + F.lam * self.perceptual_loss
            self.grad_complete_loss = tf.gradients(self.complete_loss, self.z_gen)

        # create summaries  for Tensorboard visualization
        tf.summary.scalar('disc_loss', self.d_loss)
        tf.summary.scalar('disc_loss_real', self.d_loss_real)
        tf.summary.scalar('disc_loss_fake', self.d_loss_fake)
        tf.summary.scalar('gen_loss', self.g_loss_actual)

        self.g_loss = tf.constant(0) 

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'D/d_' in var.name]
        self.g_vars = [var for var in t_vars if 'G/g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):
        # main method for training conditonal GAN

        global_step = tf.placeholder(tf.int32, [], name="global_step_iterations")

        learning_rate_D = tf.train.exponential_decay(F.learning_rate_D, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        learning_rate_G = tf.train.exponential_decay(F.learning_rate_G, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        # Create summaries to visualise weights
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name,  var)
        
        self.summary_op = tf.summary.merge_all()

        d_optim = tf.train.AdamOptimizer(learning_rate_D, beta1=F.beta1D)\
          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate_G, beta1=F.beta1G)\
          .minimize(self.g_loss_actual, var_list=self.g_vars)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

        start_time = time.time()

        if F.load_chkpt:
            try:
                self.load(F.checkpoint_dir)
                print(" [*] Checkpoint Load Success !!!")
            except:
                print(" [!] Checkpoint Load failed !!!!")
        else:
            print(" [*] Not Loaded")

        self.ra, self.rb = -1, 1
        counter = 1
        step = 1
        idx = 1

        writer = tf.summary.FileWriter(F.log_dir, graph=tf.get_default_graph())

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            while not coord.should_stop():
                start_time = time.time()
                step += 1

                # sample a noise vector 
                sample_z_gen = np.random.uniform(
                        self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)

                # Update D network
                iters = 1
                if True: 
                    train_summary, _, dloss, errD_fake, errD_real = self.sess.run(
                            [self.summary_op, d_optim,  self.d_loss, self.d_loss_fake, self.d_loss_real],
                            feed_dict={self.z_gen: sample_z_gen, global_step: counter, self.is_training: True})
                    writer.add_summary(train_summary, counter)

                # Update G network
                iters = 1  # can play around 
                if True :
                    for iter_gen in range(iters):
                        sample_z_gen = np.random.uniform(self.ra, self.rb,
                            [F.batch_size, F.z_dim]).astype(np.float32)
                        _,  gloss, dloss = self.sess.run(
                            [g_optim,  self.g_loss_actual, self.d_loss],
                            feed_dict={self.z_gen: sample_z_gen, global_step: counter, self.is_training: True})
                       
                lrateD = learning_rate_D.eval({global_step: counter})
                lrateG = learning_rate_G.eval({global_step: counter})

                print(("Iteration: [%6d] lrateD:%.2e lrateG:%.2e d_loss_f:%.8f d_loss_r:%.8f " +
                      "g_loss_act:%.8f")
                      % (idx, lrateD, lrateG, errD_fake, errD_real, gloss))

                # peridically save generated images with corresponding checkpoints

                if np.mod(counter, F.sampleInterval) == 0:
                    sample_z_gen = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
                    samples, key_pts,  d_loss, g_loss_actual = self.sess.run(
                        [self.G, self.keypoints,  self.d_loss, self.g_loss_actual],
                        feed_dict={self.z_gen: sample_z_gen, global_step: counter, self.is_training: False}
                    )
                    save_images(samples, [8, 8],
                                F.sample_dir + "/sample_" + str(counter) + ".png")
                    save_images(key_pts,  [8, 8],  
                                F.sample_dir + "/sampleky_" + str(counter) + ".png")
                    print("new samples stored!!")
                 
                # periodically save checkpoints for future loading
                if np.mod(counter, F.saveInterval) == 0:
                    self.save(F.checkpoint_dir, counter)
                    print("Checkpoint saved successfully !!!")

                counter += 1
                idx += 1
                
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)

    def poisson_blend(self, imgs1, imgs2, mask):
        # call this while performing correctness experiment
        out = np.zeros(imgs1.shape)

        for i in range(0, len(imgs1)):
            img1 = (imgs1[i] + 1.) / 2.0
            img2 = (imgs2[i] + 1.) / 2.0
            out[i] = np.clip((poissonblending.blend(img1, img2, 1 - mask) - 0.5) * 2, -1.0, 1.0)

        return out.astype(np.float32)

    def poisson_blend2(self, imgs1, imgs2, mask):
        # call this while performing consistency experiment
        out = np.zeros(imgs1.shape)

        for i in range(0, len(imgs1)):
            img1 = (imgs1[i] + 1.) / 2.0
            img2 = (imgs2[i] + 1.) / 2.0
            out[i] = np.clip((poissonblending.blend(img1, img2, 1 - mask[i]) - 0.5) * 2, -1.0, 1.0)

        return out.astype(np.float32)

    def get_psnr(self, img_true, img_gen):
        return compare_psnr(img_true.astype(np.float32), img_gen.astype(np.float32))

    def get_mse(self, img_true, img_gen):
        return compare_mse(img_true.astype(np.float32), img_gen.astype(np.float32))

    def same_z_diff_keypoints(self):
        # call this while trying to show generated samples for same z-vector but different 
        # keypoint maps

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(F.checkpoint_dir)
        assert(isLoaded)

        files = os.listdir('samples_complete/')
        imgs = [x for x in files if 'im' in x]
        keys = [x.replace('img', 'ky') for x in imgs][:64]

        for i in range(200):
            shuffle(files)
            imgs = [x for x in files if 'im' in x]
            keys = [x.replace('img', 'ky') for x in imgs][:64]
            z_new = np.random.uniform(-1, 1, size=(F.z_dim))
            batch_keypoints = np.array([get_image('samples_complete/' + batch_file, F.output_size, is_crop=self.is_crop)
                         for batch_file in keys]).astype(np.float32)

            fd = {
                self.z_gen: [z_new] * F.batch_size,
                self.keypoints: batch_keypoints,
                self.is_training: False
            }
            G_imgs = self.sess.run(self.G, feed_dict=fd)

            save_images(G_imgs, [8, 8], 'experiments/same_z_diff_k_image_' + str(i) + '.png')
            save_images(batch_keypoints, [8, 8], 'experiments/same_z_diff_k_keypts_' + str(i) + '.png')

    def diff_z_same_keypoints(self):
        # call this while trying to show generated samples for different z-vectors but
        # same keypoint map

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(F.checkpoint_dir)
        assert(isLoaded)

        files = os.listdir('samples_complete/')
        imgs = [x for x in files if 'im' in x]
        keys = [x.replace('img', 'ky') for x in imgs][:64]
  
        batch_keypoints = np.array([get_image('samples_complete/' + batch_file, F.output_size, is_crop=self.is_crop)
                     for batch_file in keys]).astype(np.float32)

        for i in range(200):
            z_new = np.random.uniform(-1, 1, size=(F.batch_size, F.z_dim))
            keypoints = np.array([batch_keypoints[(i * 7) % 64]] * F.batch_size)
            fd = {
                self.z_gen: z_new,
                self.keypoints: keypoints,
                self.is_training: False
            }
            G_imgs = self.sess.run(self.G, feed_dict=fd)

            save_images(G_imgs, [8, 8], 'experiments/diff_z_same_k_image_' + str(i) + '.png')
            save_images(keypoints, [8, 8], 'experiments/diff_z_same_k_keypts_' + str(i) + '.png')  

    def complete(self):
        # this is main method which does inpainting (correctness experiment)
        def make_dir(name):
            # Works on python 2.7, where exist_ok arg to makedirs isn't available.
            p = os.path.join(F.outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)

        make_dir('hats_imgs')
        make_dir('completed')
        make_dir('logs')

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(F.checkpoint_dir)
        assert(isLoaded)

        files = os.listdir('samples_complete/') #path of held out images for inpainitng experiment
        print("Total files to inpaint :", len(files))
        imgs = [x for x in files if 'img' in x]
        keys = [x.replace('img', 'ky') for x in imgs]
        nImgs = len(imgs)

        batch_idxs = int(np.ceil(nImgs / F.batch_size))
        print("Total batches:::", batch_idxs)
        if F.maskType == 'random':
            fraction_masked = F.fraction_masked
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0

        elif F.maskType == 'center':
            assert(F.centerScale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = F.output_size
            l = int(F.output_size * F.centerScale)
            u = int(F.output_size * (1.0-F.centerScale))
            mask[l:u, l:u, :] = 0.0

        elif F.maskType == 'left':
            mask = np.ones(self.image_shape)
            c = F.output_size // 2
            mask[:,:c,:] = 0.0
        
        elif F.maskType == 'freehand_poly':
            image = np.ones(self.image_shape)
            mask = np.ones(self.image_shape)
            if F.output_size == 128:
                contours = 2 * np.array([ [10,10], [10, 15], [7, 30], [12, 54], [35, 50], [50, 48], [30, 25]])
            else:
                contours = np.array([ [10,10], [10, 15], [7, 30], [12, 54], [35, 50], [50, 48], [30, 25]])
            black = (0, 0, 0)
            cv2.fillPoly(image, pts = [contours], color = (0, 0, 0))
            mask = image 



        elif F.maskType == 'full':
            mask = np.ones(self.image_shape)

        elif F.maskType == 'grid':
            mask = np.zeros(self.image_shape)
            mask[::4,::4,:] = 1.0

        elif F.maskType == 'checkboard':
            if F.output_size == 128:
                check_size = 64
            else:
                check_size = 32
            num_tiles = int(self.image_shape[0] / (2 * check_size))
            w1 = np.ones((check_size, check_size, 3))
            b1 = np.zeros((check_size, check_size, 3))
            stack1 = np.hstack((w1, b1))
            stack2 = np.hstack((b1, w1))
            atom = np.vstack((stack1, stack2))
            mask = np.tile(atom, (num_tiles, num_tiles, 1))

        else:
            assert(False)

        img_data_path = 'samples_complete/'

        psnr_list, psnr_list2 = [], []
        for idx in xrange(0, batch_idxs):
            print("Processing batch number:  ", idx)
            l = idx * F.batch_size
            u = min((idx + 1) * F.batch_size, nImgs)
            batchSz = u - l
            batch_files = imgs[l:u]
            batch_images = np.array([get_image(img_data_path + batch_file, F.output_size, is_crop=self.is_crop)
                                   for batch_file in batch_files]).astype(np.float32)
            
            batch_files = keys[l:u]
            batch_keypoints = np.array([get_image(img_data_path + batch_file, F.output_size, is_crop=self.is_crop)
                                      for batch_file in batch_files]).astype(np.float32)

            if batchSz < F.batch_size:
                padSz = ((0, int(F.batch_size - batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            zhats = np.random.uniform(-1, 1, size=(F.batch_size, F.z_dim))
            m = 0
            v = 0

            nRows = np.ceil(batchSz / 8)
            nCols = min(8, batchSz)
            save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(F.outDir, 'before_' + str(idx) + '.png'))
            masked_images = np.multiply(batch_images, mask)# - np.multiply(np.ones(batch_images.shape), 1.0 - mask)
            save_images(np.array(masked_images - np.multiply(np.ones(batch_images.shape), 1.0 - mask)), [nRows,nCols],
                        os.path.join(F.outDir, 'mask_' + str(idx) + '.png'))

            for i in xrange(F.nIter):
                fd = {
                    self.z_gen: zhats,
                    self.mask: [mask] * F.batch_size,
                    self.keypoints: batch_keypoints,
                    self.images: batch_images,
                    self.is_training: False
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                for img in range(batchSz):
                    with open(os.path.join(F.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
                        f.write('{} {} '.format(i, loss[img]).encode())
                        np.savetxt(f, zhats[img:img+1])

                if i % F.outInterval == 0:
                    print("Iteration: {:04d} |  Loss = {:.6f}".format(i, np.mean(loss[0:batchSz])))

                    inv_masked_hat_images = masked_images + np.multiply(G_imgs, 1.0-mask)
                    completed = inv_masked_hat_images
                    imgName = os.path.join(F.outDir,
                                           'completed/_{:02d}_{:04d}.png'.format(idx, i))
                    # scipy.misc.imsave(imgName, (G_imgs[0] + 1) * 127.5)
                    save_images(completed[:batchSz,:,:,:], [nRows,nCols], imgName)

                if F.approach == 'adam':
                    # Optimize single completion with Adam
                    m_prev = np.copy(m)
                    v_prev = np.copy(v)
                    m = F.beta1 * m_prev + (1 - F.beta1) * g[0]
                    v = F.beta2 * v_prev + (1 - F.beta2) * np.multiply(g[0], g[0])
                    m_hat = m / (1 - F.beta1 ** (i + 1))
                    v_hat = v / (1 - F.beta2 ** (i + 1))
                    zhats += - np.true_divide(F.lr * m_hat, (np.sqrt(v_hat) + F.eps))
                    zhats = np.clip(zhats, -1, 1)

                elif F.approach == 'hmc':
                    # Sample example completions with HMC (not in paper)
                    zhats_old = np.copy(zhats)
                    loss_old = np.copy(loss)
                    v = np.random.randn(self.batch_size, F.z_dim)
                    v_old = np.copy(v)

                    for steps in range(F.hmcL):
                        v -= F.hmcEps/2 * F.hmcBeta * g[0]
                        zhats += F.hmcEps * v
                        np.copyto(zhats, np.clip(zhats, -1, 1))
                        loss, g, _, _ = self.sess.run(run, feed_dict=fd)
                        v -= F.hmcEps/2 * F.hmcBeta * g[0]

                    for img in range(batchSz):
                        logprob_old = F.hmcBeta * loss_old[img] + np.sum(v_old[img]**2)/2
                        logprob = F.hmcBeta * loss[img] + np.sum(v[img]**2)/2
                        accept = np.exp(logprob_old - logprob)
                        if accept < 1 and np.random.uniform() > accept:
                            np.copyto(zhats[img], zhats_old[img])

                    F.hmcBeta *= F.hmcAnneal

                else:
                    assert(False)

            inv_masked_hat_images = masked_images + np.multiply(G_imgs, 1.0 - mask)     
            for i in range(len(masked_images)):
                psnr_list2.append(self.get_psnr(batch_images[i], inv_masked_hat_images[i]))


            blended_images = self.poisson_blend(batch_images, G_imgs, mask)
            imgName = os.path.join(F.outDir, 'completed/{:02d}_blended.png'.format(idx))
            save_images(blended_images[:batchSz,:,:,:], [nRows,nCols], imgName)
            
            for i in range(len(masked_images)):
                psnr_list.append(self.get_psnr(batch_images[i], blended_images[i]))

            print("For current batch | PSNR before blending::: ",  np.mean(psnr_list2))
            print("For current batch | PSNR after blending::: ",  np.mean(psnr_list))

        print ('Final | PSNR Before Blending:: ', np.mean(psnr_list2))
        np.save(F.outDir + '/complete_psnr_after_blend.npy', np.array(psnr_list)) # For statistical testing

        print ('Final | PSNR After Blending:: ', np.mean(psnr_list))
        np.save(F.outDir + '/complete_psnr_before_blend.npy', np.array(psnr_list2)) # For statistical testing

    def create_mask(self, centerScale=0.25, temporal=True, check_size=8):
        # specifically creates random sized/designed mask for consistency experiemnts

        if F.maskType == 'freehand_poly':
            image = np.ones(self.image_shape)
            mask = np.ones(self.image_shape)
            freehand_list = []
            freehand_list.append(np.array([ [10,10], [15,10], [30,7], [54, 12], [50, 35], [48, 50], [25, 30]]))
            freehand_list.append( np.array([ [10,10], [10, 15], [7, 30], [12, 54], [35, 50], [50, 48], [30, 25]]))
            freehand_list.append(np.array([ [20,1], [20,20], [10,52], [25, 48], [48,40], [28,20], [20,1] ]))
            freehand_list.append(np.array([ [1,20], [20,20], [52,10], [48, 25], [40,48], [20, 28], [1, 20] ]))
            index = np.random.randint(0,4)

            black = (0, 0, 0)
            if F.output_size ==128:
                 cv2.fillPoly(image, pts = [2 * freehand_list[index]], color = (0, 0, 0))
            else:
                 cv2.fillPoly(image, pts = [freehand_list[index]], color = (0, 0, 0))
            mask = image 


        elif F.maskType == 'center':
            assert(centerScale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = F.output_size
            if temporal == True:
              centerScale = random.uniform(centerScale - 0.05, centerScale + 0.05)
              
            l = int(F.output_size * centerScale)
            u = int(F.output_size * (1.0-centerScale))
            mask[l:u, l:u, :] = 0.0

        elif F.maskType == 'checkboard':
            if temporal == True:
                check_size_list = [8, 16, 32]
                index = np.random.randint(0, 3)
                check_size = check_size_list[index]

            num_tiles = int(self.image_shape[0] / (2 * check_size))
            w1 = np.ones((check_size, check_size, 3))
            b1 = np.zeros((check_size, check_size, 3))
            stack1 = np.hstack((w1, b1))
            stack2 = np.hstack((b1, w1))
            atom = np.vstack((stack1, stack2))
            mask = np.tile(atom, (num_tiles, num_tiles, 1))


 
        elif F.maskType == 'left':
            mask = np.ones(self.image_shape)
            c = F.output_size // 2
            mask[:,:c,:] = 0.0
        
        else:
            assert(False)
        return mask

    def temporal_consistency(self):
        # main method for performing experiments related to consistency
        #idea: in a batch of 64 images, there will be 8 subjects with 8 different kinds of damages
        # 8 rows of different subjects and 8 columns for each subject


        def make_dir(name):
            # Works on python 2.7, where exist_ok arg to makedirs isn't available.
            p = os.path.join(F.outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)
        make_dir('hats_imgs')
        make_dir('completed')
        make_dir('logs')

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(F.checkpoint_dir)
        assert(isLoaded)

        files = os.listdir('samples_complete/')
        imgs = [x for x in files if 'im' in x]
        keys = [x.replace('img', 'ky') for x in imgs]
        nImgs = len(imgs)
        batch_idxs = int(np.ceil(nImgs / F.batch_size))

        masks = []
        for i in range(int(F.batch_size / 8)):
            masks.append(self.create_mask(temporal=True))

        mask = np.zeros([F.batch_size, F.output_size, F.output_size, 3])
        for i in range(F.batch_size):
            mask[i] = masks[i % 8]

        img_data_path = 'samples_complete/'
        psnr_list, psnr_list2 = [], []
        for idx in xrange(0, batch_idxs * 8):  # because in a batch we are taking 8 people instead of 64
            print("Processing batch {:03d} out of {:03d}".format(idx,  batch_idxs * 8))
            batch_size = int(F.batch_size / 8)
            batchSz = F.batch_size
            l = idx * batch_size
            u = min((idx + 1) * batch_size, nImgs)
            batch_files = imgs[l:u]
            batch_images = np.array([get_image(img_data_path + batch_files[int(i / 8)], F.output_size, is_crop=self.is_crop)
                     for i in range(len(batch_files) * 8)]).astype(np.float32)

            batch_files = keys[l:u]
            batch_keypoints = np.array([get_image(img_data_path + batch_files[int(i / 8)], F.output_size, is_crop=self.is_crop)
                     for i in range(len(batch_files) * 8)]).astype(np.float32)

            nRows = np.ceil(batchSz / 8)
            nCols = min(8, batchSz)

            if batchSz < F.batch_size:
                print(batchSz)
                padSz = ((0, int(F.batch_size - batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            zhats = np.random.uniform(-1, 1, size=(F.batch_size, F.z_dim))
            m = 0
            v = 0

            nRows = np.ceil(batchSz / 8)
            nCols = min(8, batchSz)
            save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(F.outDir, 'before_' + str(idx) + '.png'))

            masked_images = np.multiply(batch_images, mask) 
            save_images(np.array(mask - np.multiply(np.ones(batch_images.shape), 1.0 - mask)), [nRows,nCols],
                        os.path.join(F.outDir, 'mask_' + str(idx) + '.png'))

            for i in xrange(F.nIter):
                fd = {
                    self.z_gen: zhats,
                    self.mask: mask,
                    self.keypoints: batch_keypoints,
                    self.images: batch_images,
                    self.is_training: False
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                for img in range(batchSz):
                    with open(os.path.join(F.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
                        f.write('{} {} '.format(i, loss[img]).encode())
                        np.savetxt(f, zhats[img:img+1])

                if i % F.outInterval == 0:
                    print("Iterations: {:04d} |  Loss = {:.4f}".format(i, np.mean(loss[0:batchSz])))

                    inv_masked_hat_images = masked_images + np.multiply(G_imgs, 1.0-mask)
                    completed = inv_masked_hat_images
                    imgName = os.path.join(F.outDir, 'completed/{:02d}_{:04d}.png'.format(idx, i))
                    save_images(completed[:batchSz,:,:,:], [nRows,nCols], imgName)

                if F.approach == 'adam':
                    # Optimize single completion with Adam
                    m_prev = np.copy(m)
                    v_prev = np.copy(v)
                    m = F.beta1 * m_prev + (1 - F.beta1) * g[0]
                    v = F.beta2 * v_prev + (1 - F.beta2) * np.multiply(g[0], g[0])
                    m_hat = m / (1 - F.beta1 ** (i + 1))
                    v_hat = v / (1 - F.beta2 ** (i + 1))
                    zhats += - np.true_divide(F.lr * m_hat, (np.sqrt(v_hat) + F.eps))
                    zhats = np.clip(zhats, -1, 1)

                elif F.approach == 'hmc':
                    # Sample example completions with HMC (not in paper)
                    zhats_old = np.copy(zhats)
                    loss_old = np.copy(loss)
                    v = np.random.randn(self.batch_size, F.z_dim)
                    v_old = np.copy(v)

                    for steps in range(F.hmcL):
                        v -= F.hmcEps/2 * F.hmcBeta * g[0]
                        zhats += F.hmcEps * v
                        np.copyto(zhats, np.clip(zhats, -1, 1))
                        loss, g, _, _ = self.sess.run(run, feed_dict=fd)
                        v -= F.hmcEps/2 * F.hmcBeta * g[0]

                    for img in range(batchSz):
                        logprob_old = F.hmcBeta * loss_old[img] + np.sum(v_old[img]**2)/2
                        logprob = F.hmcBeta * loss[img] + np.sum(v[img]**2)/2
                        accept = np.exp(logprob_old - logprob)
                        if accept < 1 and np.random.uniform() > accept:
                            np.copyto(zhats[img], zhats_old[img])

                    F.hmcBeta *= F.hmcAnneal

                else:
                    assert(False)

            inv_masked_hat_images = masked_images + np.multiply(G_imgs, 1.0 - mask)
            blended_images = self.poisson_blend2(batch_images, G_imgs, mask)
            imgName = os.path.join(F.outDir, 'completed/{:02d}_blended.png'.format(idx))
            save_images(blended_images[:batchSz,:,:,:], [nRows,nCols], imgName)

            # calculate consistency for each possible pairwise inpainted images before/after blending
            for i in range(8):
                for j in range(8):
                    for k in range(j):
                        psnr_list2.append(self.get_mse(batch_images[i * 8 + j], inv_masked_hat_images[i * 8 + k]))
                        psnr_list.append(self.get_mse(blended_images[i * 8 + j], blended_images[i * 8 + k]))
            

            print("Uptil now | MSE Before Blending::",  np.mean(psnr_list2))
            print("Uptil now | MSE After Blending::",  np.mean(psnr_list))

        print ('Final | MSE Before Blending:: ', np.mean(psnr_list2))
        np.save(F.outDir + '/mse_before_blend.npy', np.array(psnr_list))   # for statistical testing

        print ('Final | MSE After Blending:: ', np.mean(psnr_list))
        np.save(F.outDir + '/mse_after_blend.npy', np.array(psnr_list2))  # for statistical testing

    def discriminator(self, image, keypoints, reuse=False):
        with tf.variable_scope('D'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            dim = 64
            image = tf.concat([image, keypoints], 3)
            if F.output_size == 128:
                  h0 = lrelu(conv2d(image, dim, name='d_h0_conv'))
                  h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim * 2, name='d_h1_conv')))
                  h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim * 4, name='d_h2_conv')))
                  h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, dim * 8, name='d_h3_conv')))
                  h4 = lrelu(batch_norm(name='d_bn4')(conv2d(h3, dim * 16, name='d_h4_conv')))
                  h4 = tf.reshape(h4, [F.batch_size, -1])
                  h5 = linear(h4, 1, 'd_h5_lin')
                  return tf.nn.sigmoid(h5), h5

            else:
                  h0 = lrelu(conv2d(image, dim, name='d_h0_conv'))
                  h1 = lrelu(batch_norm(name='d_bn1')(conv2d(h0, dim * 2, name='d_h1_conv')))
                  h2 = lrelu(batch_norm(name='d_bn2')(conv2d(h1, dim * 4, name='d_h2_conv')))
                  h3 = lrelu(batch_norm(name='d_bn3')(conv2d(h2, dim * 8, name='d_h3_conv')))
                  h4 = tf.reshape(h3, [F.batch_size, -1])
                  h5 = linear(h4, 1, 'd_h5_lin')
                  return tf.nn.sigmoid(h5), h5

    def generator(self, z, keypoints):
        dim = 64
        k = 5
        with tf.variable_scope("G"):
              s2, s4, s8, s16 = int(F.output_size / 2), int(F.output_size / 4), int(F.output_size / 8), int(F.output_size / 16)
              z = tf.reshape(z, [F.batch_size, 1, 1, 100])
              z = tf.tile(z, [1, F.output_size, F.output_size, 1])
              z = tf.concat([z, keypoints], 3)

              h0 = z
            
              h1 = tf.nn.relu(batch_norm(name='g_bn1')(conv2d(h0, dim * 2, 5, 5, 1, 1, name='g_h1_conv'), self.is_training))
              h2 = tf.nn.relu(batch_norm(name='g_bn2')(conv2d(h1, dim * 2, k, k, 2, 2, name='g_h2_conv'), self.is_training))
              h3 = tf.nn.relu(batch_norm(name='g_bn3')(conv2d(h2, dim * 4, k, k, 2, 2, name='g_h3_conv'), self.is_training))
              h4 = tf.nn.relu(batch_norm(name='g_bn4')(conv2d(h3, dim * 8, k, k, 2, 2, name='g_h4_conv'), self.is_training))
              h5 = tf.nn.relu(batch_norm(name='g_bn5')(conv2d(h4, dim * 16, k, k, 2, 2, name='g_h5_conv'), self.is_training))

              h6 = deconv2d(h5, [F.batch_size, s8, s8, dim * 8], k, k, 2, 2, name = 'g_deconv1')
              h6 = tf.nn.relu(batch_norm(name = 'g_bn6')(h6, self.is_training))
                      
              h7 = deconv2d(h6, [F.batch_size, s4, s4, dim * 4], k, k, 2, 2, name = 'g_deconv2')
              h7 = tf.nn.relu(batch_norm(name = 'g_bn7')(h7, self.is_training))

              h8 = deconv2d(h7, [F.batch_size, s2, s2, dim * 2], k, k, 2, 2, name = 'g_deconv4')
              h8 = tf.nn.relu(batch_norm(name = 'g_bn8')(h8, self.is_training))

              h9 = deconv2d(h8, [F.batch_size, F.output_size, F.output_size, 3], k, k, 2, 2, name ='g_hdeconv5')
              h9 = tf.nn.tanh(h9, name = 'g_tanh')
              return h9
              
    def save(self, checkpoint_dir, step=0):
        model_name = "model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
