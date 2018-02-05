
# Consistent_Inpainting_GAN
![ ](imgs/front.PNG?raw=true "Semantically guided GAN for face generation")
This is the training code of our paper on consistent semantic inpainting of faces in Tensorflow | [Paper](http://arxiv.org/pdf/1711.06106.pdf).
We show that conditioning GANs with facial semantic maps helps in :
+ Better image generation capability of generator
+ Better PSNR and visual performance during inpainting
+ Decoupling of pose and appearance of face
+ Consistency in inpainting
+ We improve upon [DIP](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.pdf), CVPR-2017


### Dependencies
+ Tensorflow >= 1.0
+ Dlib (for facial keypoint detection)
+ pyamg (for Poisson Blending)

### Preprocessing
+ Download the celebA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and move 2560 images to `data/test/images` and others to `data/train/images`.
+ Download and extract the trained facial shape predictor from [here]( http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
+ Run the script to create tfrecords for training the model: 

    `python preprocess_train_images.py shape_predictor_68_face_landmarks.dat`
+ Run the script to generate keypoints maps for test images: 

    `python preprocess_test_images.py shape_predictor_68_face_landmarks.dat`

### Training
+ For training the model:

  `python main.py --batch_size=64 --output_size=128`
+ The generated samples with the facial keypoint maps are saved in `samples/celebA`.
+ To run the completeness experiment:

  `python complete.py --batch_size=64 --output_size=128`
+ To run the consistency experiment:

  `python temporal.py --batch_size=64 --output_size=128`
  
### Independence of Pose and Appearance
We show that conditioning GANs with facial maps helps in decoupling apperance of face(skin textures, gender) from pose (scale, orientation, facial global expression)
+ Different `z` vector but same facial maps
![ ](imgs/same_k_diff_z.PNG?raw=true "Semantically guided GAN for face generation")

+ Different facial maps but same `z` vector
![ ](imgs/same_z_diff_k.PNG?raw=true "Semantically guided GAN for face generation")

  
### Visualization of Consistency
We evaluated "consistency" on pseudo corrupted sequences. 
+ Given an original starting image, corrupt it with different masks
+ Inpaint the corrupted images
+ Ideally all reconstructions should be identical
+ Parwise MSE between inpainted images gives a measure of consistency (Refer to paper for metrics)

![ ](imgs/64X64_sequence_1.gif?raw=true "Semantically guided GAN for face generation") ![ ](imgs/64X64_sequence_2.gif?raw=true "Semantically guided GAN for face generation") ![ ](imgs/64X64_sequence_3.gif?raw=true "Semantically guided GAN for face generation")

![ ](imgs/128X128_sequence_3.gif?raw=true "Semantically guided GAN for face generation")



### Citation
If you find our work useful in your research, please cite:

    @article{lahiri2017improving,
        title={Improving Consistency and Correctness of Sequence Inpainting using Semantically Guided Generative Adversarial Network},
        author={Lahiri, Avisek and Jain, Arnav and Biswas, Prabir Kumar and Mitra, Pabitra},
        journal={arXiv preprint arXiv:1711.06106},
        year={2017}
    }
