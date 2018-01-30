# error_concealment_GAN
error concealment in videophone sequences using GANs

### How to create the facial semantic maps
+ Download the celebA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and move 2500 images to test/images and others in train/images
+ Create new directories named train_records/ and test_records/
+ Download and extract the trained facial shape predictor from [here]( http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
+ Run the script to create tfrecords of facial maps of test and train images: $convert_to_records.py shape_predictor_68_face_landmarks.dat train/test
