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

class DCGAN(object):
    def __init__(self, sess):
        self.sess = sess
        if F.dataset != "lsun" and F.inc_score:
            print("Loading inception module")
            self.inception_module = inception_score(self.sess)
            print("Inception module loaded")

        self.image_shape = [F.output_size, F.output_size, 3]
        self.build_model()
        self.is_crop = False

    def build_model(self):
        if F.use_tfrecords == True:
            # load images from tfrecords + queue thread runner for better GPU utilization
            tfrecords_filename = ['train_data/' + x for x in os.listdir('train_data/')]
            filename_queue = tf.train.string_input_producer(
                                tfrecords_filename, num_epochs=100)


            self.images, _ = reader.read_and_decode(filename_queue, F.batch_size)

            if F.output_size == 64:
                self.images = tf.image.resize_images(self.images, (64, 64))

            self.images = (self.images / 127.5) - 1

        else:    
            self.images = tf.placeholder(tf.float32,
                                       [F.batch_size, F.output_size, F.output_size,
                                        F.c_dim],
                                       name='real_images')
        
        self.z_gen = tf.placeholder(tf.float32, [None, F.z_dim], name='z')        

        self.G_mean = self.generator(self.z_gen)
        self.D, self.D_logits = self.discriminator(self.images, reuse=False)
        self.D_, self.D_logits_, = self.discriminator(self.G_mean, reuse=True)

        #calculations for getting hard predictions
        # +1 means fooled the D network while -1 mean D has won
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits, labels = tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits_, labels = tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_actual = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits_, labels = tf.ones_like(self.D_)))

        self.g_loss = tf.constant(0)        

        if F.error_conceal == True:
            self.mask = tf.placeholder(tf.float32, [F.batch_size] + self.image_shape, name='mask')
            self.contextual_loss = tf.reduce_sum(
              tf.contrib.layers.flatten(
              tf.abs(tf.multiply(self.mask, self.G_mean) - tf.multiply(self.mask, self.images))), 1)
            self.perceptual_loss = self.g_loss_actual
            self.complete_loss = self.contextual_loss + F.lam * self.perceptual_loss
            self.grad_complete_loss = tf.gradients(self.complete_loss, self.z_gen)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):
        """Train DCGAN"""
        data = dataset()
        global_step = tf.placeholder(tf.int32, [], name="global_step_epochs")

        

        learning_rate_D = tf.train.exponential_decay(F.learning_rate_D, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        learning_rate_G = tf.train.exponential_decay(F.learning_rate_G, global_step,
                                                     decay_steps=F.decay_step,
                                                     decay_rate=F.decay_rate, staircase=True)
        d_optim = tf.train.AdamOptimizer(learning_rate_D, beta1=F.beta1D)\
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate_G, beta1=F.beta1G)\
            .minimize(self.g_loss_actual, var_list=self.g_vars)

        tf.initialize_all_variables().run() 

        counter = 0
        start_time = time.time()

        if F.load_chkpt:
            try:
                self.load(F.checkpoint_dir)
                print(" [*] Load SUCCESS")
            except:
                print(" [!] Load failed...")
        else:
            print(" [*] Not Loaded")

        self.ra, self.rb = -1, 1

        for epoch in xrange(1000000):
            idx = 0
            iscore = 0.0, 0.0 # self.get_inception_score()
            batch_iter = data.batch()
            for sample_images in batch_iter:
                
                sample_z_gen = np.random.uniform(
                    self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
		
		errG_actual = self.g_loss_actual.eval({self.z_gen: sample_z_gen})
		E_Fake = self.d_loss_fake.eval({self.z_gen: sample_z_gen})
                E_Real = self.d_loss_real.eval({self.images: sample_images})

		
                # Update D network
                iters = 1
                if True: 
                  #print('Train D net')
                  _,  dlossf = self.sess.run(
                      [d_optim,  self.d_loss_fake],
                      feed_dict={self.images: sample_images, 
                                self.z_gen: sample_z_gen, global_step: epoch})
                  
                 

                # Update G network
                iters = 1
                if True :
                   sample_z_gen = np.random.uniform(self.ra, self.rb,
		   				[F.batch_size, F.z_dim]).astype(np.float32)
                   #print('Train G Net')
                   _,  gloss, dloss = self.sess.run(
                        [g_optim,  self.g_loss, self.d_loss],
                        feed_dict={self.images: sample_images, self.z_gen: sample_z_gen,
                                    global_step: epoch})
                   

                errD_fake = self.d_loss_fake.eval({ self.z_gen: sample_z_gen})
                errD_real = self.d_loss_real.eval({self.images: sample_images})
                errG = self.g_loss.eval({self.z_gen: sample_z_gen})
                errG_actual = self.g_loss_actual.eval({self.z_gen: sample_z_gen})
                lrateD = learning_rate_D.eval({global_step: epoch})
                lrateG = learning_rate_G.eval({global_step: epoch})
                

                counter += 1
                idx += 1
                print(("Epoch:[%2d] [%4d/%4d]  d_loss_f:%.8f d_loss_r:%.8f " +
                      "g_loss_act:%.2f ")
                      % (epoch, idx, data.num_batches,  errD_fake,
                         errD_real, errG_actual))



                if np.mod(counter, 500) == 1:
                    sample_z_gen = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
                    samples, d_loss, g_loss_actual = self.sess.run(
                        [self.G_mean, self.d_loss, self.g_loss_actual],
                        feed_dict={self.z_gen: sample_z_gen, self.images: sample_images}
                    )
                    save_images(samples, [8, 8],
                                F.sample_dir + "/sample.png")
                    print("samples saved")

                if np.mod(counter, 500) == 1:
                    self.save(F.checkpoint_dir)
                    print("Checkpoint saved")
            
            sample_z_gen = np.random.uniform(self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
            samples, d_loss, g_loss_actual = self.sess.run(
                [self.G_mean, self.d_loss, self.g_loss_actual],
                feed_dict={ self.z_gen: sample_z_gen, self.images: sample_images}
            )
            save_images(samples, [8, 8],
                        F.sample_dir + "/train_{:03d}.png".format(epoch))
            #if epoch % 5 == 0:
            #    iscore = self.get_inception_score()

        # imgs2 is generator output

    # imgs2 is generator output
    def poisson_blend(self, imgs1, imgs2, mask):
        out = np.zeros(imgs1.shape)

        for i in range(0, len(imgs1)):
            img1 = (imgs1[i] + 1.) / 2.0
            img2 = (imgs2[i] + 1.) / 2.0
            out[i] = np.clip((poissonblending.blend(img1, img2, 1 - mask) - 0.5) * 2, -1.0, 1.0)
            # print (np.max(out[i]), np.min(out[i]))

        return out.astype(np.float32)

    # imgs2 is generator output
    def poisson_blend2(self, imgs1, imgs2, mask):
        out = np.zeros(imgs1.shape)

        for i in range(0, len(imgs1)):
            #print 'here',np.max(imgs1[i]), np.max(imgs2[i]), np.min(imgs1[i]), np.min(imgs2[i]) 
            img1 = (imgs1[i] + 1.) / 2.0
            img2 = (imgs2[i] + 1.) / 2.0
            out[i] = np.clip((poissonblending.blend(img1, img2, 1 - mask[i]) - 0.5) * 2, -1.0, 1.0)
            # print (np.max(out[i]), np.min(out[i]))

        return out.astype(np.float32)

    def get_psnr(self, img_true, img_gen):
        return compare_psnr(img_true.astype(np.float32), img_gen.astype(np.float32))

    def get_mse(self, img_true, img_gen):
        return compare_mse(img_true.astype(np.float32), img_gen.astype(np.float32))
        
    def complete(self):
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

        files = os.listdir('/home/avisek/faces_angie/')    #os.listdir('samples_complete/')
        imgs = [x for x in files if 'face' in x]
        nImgs = len(imgs)
        print ('Number of images is', nImgs)

        batch_idxs = int(np.ceil(nImgs / F.batch_size))

        
        # lowres_mask = np.zeros(self.lowres_shape)
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

        elif F.maskType == 'left':
            mask = np.ones(self.image_shape)
            c = F.output_size // 2
            mask[:,:c,:] = 0.0
        
        elif F.maskType == 'freehand_poly':
            image = np.ones(self.image_shape)
            mask = np.ones(self.image_shape)
            #points = [[(10,10), (23,23), (31,31), (40,40)]]
            #roi_corners = np.array(points)
            #contours = np.array( [ [32,32],[16,48], [48,16], [48,48], [30,30], [20,20], [10,50] ] )
            #contours = np.array([ [10,10], [15,10], [30,7], [54, 12], [50, 35], [48, 50], [25, 30]           ])
            if F.output_size == 128:
                contours = 2 * np.array([ [10,10], [10, 15], [7, 30], [12, 54], [35, 50], [50, 48], [30, 25]])
            else:
                contours = np.array([ [10,10], [10, 15], [7, 30], [12, 54], [35, 50], [50, 48], [30, 25]           ])

            black = (0, 0, 0)
            cv2.fillPoly(image, pts = [contours], color = (0, 0, 0))
            mask = image #np.logical_and(image, mask)
            #mask = cv2.bitwise_and(image, mask) 


        else:
            assert(False)
        
        img_data_path = '/home/avisek/faces_angie/'  #'samples_complete/'
        psnr_list, psnr_list2 = [], []

        for idx in xrange(0, batch_idxs):
            l = idx * F.batch_size
            u = min((idx + 1) * F.batch_size, nImgs)
            batchSz = u - l
            batch_files = imgs[l:u]
            #print(batch_files)
            batch_images = np.array([get_image(img_data_path + batch_file, F.output_size, is_crop=self.is_crop)
                     for batch_file in batch_files]).astype(np.float32)
            #print np.mean((scipy.misc.imread(img_data_path + batch_files[0]) / 127.5) - 1), np.mean(batch_images[0])
            #print("mean is ::", batchSz,  F.batch_size,  np.mean(batch_images),  np.max(batch_images),  np.min(batch_images))
            
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
            #sys.exit()
            masked_images = np.multiply(batch_images, mask)# - np.multiply(np.ones(batch_images.shape), 1.0 - mask)
            save_images(np.array(masked_images - np.multiply(np.ones(batch_images.shape), 1.0 - mask)), [nRows,nCols],
                        os.path.join(F.outDir, 'mask_' + str(idx) + '.png'))

            for i in xrange(F.nIter):
                fd = {
                    self.z_gen: zhats,
                    self.mask: [mask] * F.batch_size,
                    self.images: batch_images,
                    # self.is_training: False
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G_mean]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                for img in range(batchSz):
                    with open(os.path.join(F.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
                        f.write('{} {} '.format(i, loss[img]).encode())
                        np.savetxt(f, zhats[img:img+1])

                if i % F.outInterval == 0:
                    print(i, np.mean(loss[0:batchSz]))
                    # imgName = os.path.join(F.outDir,
                    #                        'hats_imgs/{:04d}.png'.format(i))
                    # nRows = np.ceil(batchSz/8)
                    # nCols = min(8, batchSz)
                    # save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)
                    # if lowres_mask.any():
                    #     imgName = imgName[:-4] + '.lowres.png'
                    #     save_images(np.repeat(np.repeat(lowres_G_imgs[:batchSz,:,:,:],
                    #                           self.lowres, 1), self.lowres, 2),
                    #                 [nRows,nCols], imgName)

                    inv_masked_hat_images = masked_images + np.multiply(G_imgs, 1.0-mask)
                    # print ('debug 1: ', np.mean(mask), np.max(inv_masked_hat_images), np.min(G_imgs), np.max(masked_images), np.min(masked_images))
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
            imgName = os.path.join(F.outDir,
                                           'completed/{:02d}_blended.png'.format(idx))
                    # scipy.misc.imsave(imgName, (G_imgs[0] + 1) * 127.5)
            save_images(blended_images[:batchSz,:,:,:], [nRows,nCols], imgName)
            
            for i in range(len(masked_images)):
                psnr_list.append(self.get_psnr(batch_images[i], blended_images[i]))
            
            print("For current batch | PSNR before blending::: ",  np.mean(psnr_list2))
            print("For current batch | PSNR after blending::: ",  np.mean(psnr_list))
        
        np.save(F.outDir + '/complete_psnr_vals.npy', np.array(psnr_list))
        np.save(F.outDir + '/complete_psnr_vals_noblend.npy', np.array(psnr_list2))


        print ('Final | PSNR Before Blending:: ', np.mean(psnr_list2))
        print ('Final | PSNR After Blending:: ', np.mean(psnr_list))
    
    def create_mask(self, centerScale=0.25, temporal=False, check_size=8):

        if F.maskType == 'freehand_poly':
            image = np.ones(self.image_shape)
            mask = np.ones(self.image_shape)
            freehand_list = []
            freehand_list.append(np.array([ [10,10], [15,10], [30,7], [54, 12], [50, 35], [48, 50], [25, 30]]))
            freehand_list.append( np.array([ [10,10], [10, 15], [7, 30], [12, 54], [35, 50], [50, 48], [30, 25]]))
            freehand_list.append(np.array([ [20,1], [20,20], [10,52], [25, 48], [48,40], [28,20], [20,1] ]))
            freehand_list.append(np.array([ [1,20], [20,20], [52,10], [48, 25], [40,48], [20, 28], [1, 20] ]))

            #contours = np.array( [ [32,32],[16,48], [48,16], [48,48], [30,30], [20,20], [10,50] ] )
            #contours = np.array([ [10,10], [15,10], [30,7], [54, 12], [50, 35], [48, 50], [25, 30]           ])
            #contours = np.array([ [10,10], [10, 15], [7, 30], [12, 54], [35, 50], [50, 48], [30, 25]           ])
            #contours =  np.array([ [20,1], [20,20], [10,52], [25, 48], [48,40], [28,20], [20,1] ])
            index = np.random.randint(0,4)

            black = (0, 0, 0)
            if F.output_size == 128:
                cv2.fillPoly(image, pts = [2 * freehand_list[index]], color = (0, 0, 0))
            else:
                cv2.fillPoly(image, pts = [freehand_list[index]], color = (0, 0, 0))
            mask = image #np.logical_and(image, mask)
            #mask = cv2.bitwise_and(image, mask)



        elif F.maskType == 'random':
            fraction_masked = 0.7
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0

        elif F.maskType == 'center':
            assert(centerScale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = F.output_size
            if temporal == True:
              centerScale = random.uniform(centerScale - 0.05, centerScale + 0.05)


            l = int(F.output_size * centerScale)
            u = int(F.output_size * (1.0-centerScale))
            mask[l:u, l:u, :] = 0.0

        elif F.maskType == 'left':
            mask = np.ones(self.image_shape)
            c = F.output_size // 2
            mask[:,:c,:] = 0.0

        elif F.maskType == 'full':
            mask = np.ones(self.image_shape)

        elif F.maskType == 'grid':
            mask = np.zeros(self.image_shape)
            mask[::4,::4,:] = 1.0

        elif F.maskType == 'checkboard':
            if temporal == True:
                check_size_list = [8, 16, 32]
                index = np.random.randint(0,3)
                check_size = check_size_list[index]

            num_tiles = int(self.image_shape[0] / (2 * check_size))
            w1 = np.ones((check_size, check_size, 3))
            b1 = np.zeros((check_size, check_size, 3))
            stack1 = np.hstack((w1, b1))
            stack2 = np.hstack((b1, w1))
            atom = np.vstack((stack1, stack2))
            mask = np.tile(atom, (num_tiles, num_tiles, 1))
        else:
            assert(False)
        return mask

    def temporal_consistency(self):
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
        imgs = [x for x in files if 'im' in x][:32]
        nImgs = len(imgs)
        # print ('Number of images is', nImgs)

        batch_idxs = int(np.ceil(nImgs / F.batch_size))

        masks = []
        for i in range(int(F.batch_size / 8)):
            masks.append(self.create_mask(temporal=True))

        mask = np.zeros([F.batch_size, F.output_size, F.output_size, 3])
        for i in range(F.batch_size):
            mask[i] = masks[i % 8]

        save_images(mask[:F.batch_size,:,:,:], [8,8],
                        os.path.join(F.outDir, 'mask.png'))

        img_data_path = 'samples_complete/'
        psnr_list, psnr_list2 = [], []
        for idx in xrange(0, int(batch_idxs * 8)):
            batch_size = int(F.batch_size / 8)
            batchSz = F.batch_size
            l = idx * batch_size
            u = min((idx + 1) * batch_size, nImgs)
            batch_files = imgs[l:u]
            batch_images = np.array([get_image(img_data_path + batch_files[int(i / 8)], F.output_size, is_crop=self.is_crop)
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

            masked_images = np.multiply(batch_images, mask)# - np.multiply(np.ones(batch_images.shape), 1.0 - mask)
            save_images(np.array(masked_images - np.multiply(np.ones(batch_images.shape), 1.0 - mask)), [nRows,nCols],
                        os.path.join(F.outDir, 'mask_' + str(idx) + '.png'))

            for i in xrange(F.nIter):
                fd = {
                    self.z_gen: zhats,
                    self.mask: mask,
                    self.images: batch_images,
                    # self.is_training: False
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G_mean]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                for img in range(batchSz):
                    with open(os.path.join(F.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
                        f.write('{} {} '.format(i, loss[img]).encode())
                        np.savetxt(f, zhats[img:img+1])

                if i % F.outInterval == 0:
                    print(i, np.mean(loss[0:batchSz]))
                    # imgName = os.path.join(F.outDir,
                    #                        'hats_imgs/{:04d}.png'.format(i))
                    # nRows = np.ceil(batchSz/8)
                    # nCols = min(8, batchSz)
                    # save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)
                    # if lowres_mask.any():
                    #     imgName = imgName[:-4] + '.lowres.png'
                    #     save_images(np.repeat(np.repeat(lowres_G_imgs[:batchSz,:,:,:],
                    #                           self.lowres, 1), self.lowres, 2),
                    #                 [nRows,nCols], imgName)

                    inv_masked_hat_images = masked_images + np.multiply(G_imgs, 1.0-mask)
                    # print ('debug 1: ', np.mean(mask), np.max(inv_masked_hat_images), np.min(G_imgs), np.max(masked_images), np.min(masked_images))
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

            for i in range(8):
                for j in range(8):
                    for k in range(j):
                        psnr_list2.append(self.get_mse(batch_images[i * 8 + j], inv_masked_hat_images[i * 8 + k]))

            blended_images = self.poisson_blend2(batch_images, G_imgs, mask)
            imgName = os.path.join(F.outDir,
                                           'completed/{:02d}_blended.png'.format(idx))
                    # scipy.misc.imsave(imgName, (G_imgs[0] + 1) * 127.5)
            save_images(blended_images[:batchSz,:,:,:], [nRows,nCols], imgName)
            
            for i in range(8):
                for j in range(8):
                    for k in range(j):
                        psnr_list.append(self.get_mse(blended_images[i * 8 + j], blended_images[i * 8 + k]))        

            print("Uptil now | MSE Before Blending::",  np.mean(psnr_list2))
            print("Uptil now | MSE After Blending::",  np.mean(psnr_list))
        
        np.save(F.outDir + '/temporal_psnr_vals.npy', np.array(psnr_list))
        np.save(F.outDir + '/complete_psnr_vals_noblend.npy', np.array(psnr_list2))
        print ('Final | MSE Before Blending:: ', np.mean(psnr_list2))
        print ('Final | MSE After Blending:: ', np.mean(psnr_list))
       
    def get_inception_score(self):
        if F.dataset == "lsun" or not F.inc_score:
            return 0.0, 0.0

        samples = []
        for k in range(50000 // F.batch_size):
            sample_z = np.random.uniform(
                self.ra, self.rb, [F.batch_size, F.z_dim]).astype(np.float32)
            images = self.sess.run(self.G_mean, {self.z: sample_z})
            samples.append(images)
        samples = np.vstack(samples)
        return self.inception_module.get_inception_score(samples)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope('D'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if F.dataset == "celebA":
                dim = 64   # intially it was 64 for 64X64 generator output
                # this is meant for 128x128 images
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

    def generator(self, z):
        with tf.variable_scope("G"):
            if F.dataset == "lsun" or F.dataset == "celebA":
                s = F.output_size
                dim = 64
                #s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
                s2, s4, s8, s16, s32 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s/32)
                
                if (s == 128):  # idea taken from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
                    z_ = linear(z, dim * 16 * 4* 4, scope = 'g_h0_lin')
                    h0 = tf.reshape(z_, [-1, 4, 4, dim * 16])
                    h0 = tf.nn.relu(batch_norm(name='g_bn0')(h0))

                    h1 = deconv2d(h0, [F.batch_size, 8, 8, dim * 8], name='g_h1')
                    h1 = tf.nn.relu(batch_norm(name='g_bn1')(h1))

                    h2 = deconv2d(h1, [F.batch_size, 16, 16, dim * 4], name='g_h2')
                    h2 = tf.nn.relu(batch_norm(name='g_bn2')(h2))

                    h3 = deconv2d(h2, [F.batch_size, 32, 32, dim * 2], name='g_h3')
                    h3 = tf.nn.relu(batch_norm(name='g_bn3')(h3))

                    h4 = deconv2d(h3, [F.batch_size, 64, 64, dim * 1], name='g_h4')
                    h4 = tf.nn.relu(batch_norm(name='g_bn4')(h4))

                    h5 = deconv2d(h4, [F.batch_size, 128, 128, F.c_dim], name='g_h5')
                    h5 = tf.nn.tanh(h5)
                    return h5
                else:
                    z_ = linear(z, dim * 8 * 4* 4, scope = 'g_h0_lin')
                    h0 = tf.reshape(z_, [-1, 4, 4, dim * 8])
                    h0 = tf.nn.relu(batch_norm(name='g_bn0')(h0))

                    h1 = deconv2d(h0, [F.batch_size, 8, 8, dim * 4], name='g_h1')
                    h1 = tf.nn.relu(batch_norm(name='g_bn1')(h1))

                    h2 = deconv2d(h1, [F.batch_size, 16, 16, dim * 2], name='g_h2')
                    h2 = tf.nn.relu(batch_norm(name='g_bn2')(h2))

                    h3 = deconv2d(h2, [F.batch_size, 32, 32, dim * 1], name='g_h3')
                    h3 = tf.nn.relu(batch_norm(name='g_bn3')(h3))

                    h4 = deconv2d(h3, [F.batch_size, 64, 64, F.c_dim], name='g_h4')
                    h4 = tf.nn.tanh(h4)
                    return h4

    def save(self, checkpoint_dir):
        model_name = "model.ckpt"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name))

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
