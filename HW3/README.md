# Solution to HW3

*by Alexander Goponenko*

## [Problem 1] (HW3_Problem1.ipynb)

I used more aggressive data augmentation (than in the notebook shown in the class) and dropouts for fully connected network to get better results.

Also I used preprocess_input from keras.applications.vgg19 in the ImageDataGenerator. It improved accuracy considerably.

Having a more complicated "head" seemed to be unjustified given just one binary classifier and the tendency of overfitting. I tried to add another convolutional layer, but it did not do any good.
