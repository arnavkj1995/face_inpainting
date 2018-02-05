# error_concealment_GAN
This is the training code of our paper on semantic inpainting of faces in tensorflow.

### Dependencies
+ Tensorflow >= 1.0
+ Dlib (for facial keypoint detection)
+ pyamg (for Poisson Blending)

### Preprocessing
+ Download the celebA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and move 2560 images to test/images and others in train/images
+ Download and extract the trained facial shape predictor from [here]( http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
+ Run the script to create tfrecords for training the model: 

    `preprocess_train_images.py shape_predictor_68_face_landmarks.dat`
+ Run the script to generate keypoints maps for test images: 

    `preprocess_test_images.py shape_predictor_68_face_landmarks.dat`

### Training
+ For training the model:

  `python main.py --batch_size=64 --output_size=128`
+ The generated samples with the facial keypoint maps are saved in `samples/celebA`.
+ To run the completeness experiment:

  `python complete.py --batch_size=64 --output_size=128`
+ To run the consistency experiment:

  `python temporal.py --batch_size=64 --output_size=128`

### Citation
If you find our work useful in your research, please cite:

    @article{lahiri2017improving,
        title={Improving Consistency and Correctness of Sequence Inpainting using Semantically Guided Generative Adversarial Network},
        author={Lahiri, Avisek and Jain, Arnav and Biswas, Prabir Kumar and Mitra, Pabitra},
        journal={arXiv preprint arXiv:1711.06106},
        year={2017}
    }
