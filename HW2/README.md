# Solution to HW2

*by Alexander Goponenko*

## Introduction

The goal of this homework was to investigate convolutional neural networks for the CIFAR10 data set. The study described in the forthcoming report followed rather "evolutional path" than "rational design". Therefore, the report is also somewhat chronological. Because more scientifically rigid approach would take much more time, quite subjective decisions were often made during the project. Thus, the conclusions cannot be trusted 100%.

### Overview
### Notes on implementation

To make files more concise, the output of network fitting functions was silenced and plots were created instead.

The testing set was shuffled pseudo-randomly but reproducible in order to be able to restart notebooks.

## 5x5 CNNs ([file 1](HW2_file1.ipynb))


It seemed that the complexity of the images could be too much for 3x3 CNNs. Thus I decided to check more complicated networks and then compare them with the regular, 3x3 networks.

### Notes on implementation

I couldn't test more than 2 convolutional layers without modifications to "standard" structure of CNNs. 

I tried 3 networks that differed in the structure of dense layers.

At first I split the original training set into training and validation sets 90:10, then redid for 80:20.

### Results

All networks demonstrated overfitting and couldn't reach above 70% validation accuracy.

## Simple 3x3 CNNs ([file 2](HW2_file2.ipynb))


These are the straightforward networks with "standard" structure of CNNs, which would be compared to "5x5" networks.

### Notes on implementation

With 3x3 layers, up to 3 layers are possible without modifications to "standard" structure of CNNs. 

I tried several networks (3 presented) with different number of layers and varying other parameters.

### Results

Among all networks, the network with 3 convolutional layers performed slightly better, reaching ~70% validation accuracy. It was also better than 5x5 networks.

Thus, I concluded that the depth of the network may be more beneficial than other parameters.

All networks demonstrated overfitting.

## Deeper 3x3 CNNs ([file 3](HW2_file3.ipynb))


I peeked into the literature for inspiration and found that deeper networks could be build if some "maxpooling" layers between the convolutional layers were omitted (https://arxiv.org/pdf/1409.1556.pdf).

### Notes on implementation

Even without maxpooling, each convolutional layer reduces each image dimension by 2 px. I made some effort to ensure that the maxpooling layers are not presented with odd-sized inputs (although it probably was not important).

3 representative networks are shown.

### Results

These networks did not do any better than the networks tried before. Yet, the literature suggested that there was a hope for them.

## Dropouts ([file 4](HW2_file4.ipynb))


Dropouts should be able to help fight overfitting and may get validation accuracy better.

### Notes on implementation

Best networks (probably subjective choice since the measurement errors were unknown) from files 1-2 and all networks from file 3 were chosen to try dropouts.

The number of epochs was increased from 50 to 200 to see more clearly the potential of the networks.

### Results

Dropouts helped raise accuracy to ~80%, but could not remove overfitting completely.

Network 4 (network 2 from file 3) showed somewhat better result (validation accuracy 81.5%).

## Data augmentation ([file 5](HW2_file5.ipynb) and [file 5a](HW2_file5a.ipynb))


Image augmentation was suggested in class and in https://github.com/BIGBALLON/cifar-10-cnn/blob/master/3_Vgg19_Network/Vgg19_keras.py.

### Notes on implementation

The "best network" (network 2 from file 3) was tested. I also tried to improve the "best network" and made a modification of it with more trainable parameters (but the modifications did not help)

Additionally, the learning rate was controlled with `keras.callbacks.ReduceLROnPlateau` for the rest of the project. When needed the number of epochs was increased up to 300.

The parameters for data augmentation were taken from https://github.com/BIGBALLON/cifar-10-cnn/blob/master/3_Vgg19_Network/Vgg19_keras.py. They were later compared to the parameters from "training convnet from scratch, using data augmentation and dropout" from the class (see "Fine Tuning" below)

### Data augmentation with dropouts ([file 5](HW2_file5.ipynb))

After over 200 epochs, the accuracy on the training set reached ~87.5%.

### Data augmentation without dropouts ([file 5a](HW2_file5a.ipynb))

After over 200 epochs, the accuracy on the training set reached ~84.5%.

### Conclusion

|                  |  no DA | with DA |
|------------------|:------:|:-------:|
|**no dropouts**   | 67.7%  |  84.5%  |
|**with dropouts** | 81.5%  |  87.5%  |

## Even deeper network ([file 6](HW2_file6.ipynb))

According to https://github.com/BIGBALLON/cifar-10-cnn/blob/master/3_Vgg19_Network/Vgg19_keras.py. a VGG-19 based network is able to reach 93.5% on the same dataset. Using this fact as motivation, I investigated a more deeper network. However, the approach described in https://github.com/BIGBALLON/cifar-10-cnn/blob/master/3_Vgg19_Network/Vgg19_keras.py uses several techniques which are beyond the scope of this assignment, including
* batch normalization
* regularization
* pre-training on simpler networks
* data pre-processing

Therefore, I investigated a deeper network which I could implement without these techniques, using VGG approach as an inspiration.

### Notes on implementation

The most important feature that I took from VGG approach is the use of `padding='same'` option in the convolutional layers. With this option, the dimensions of the "image" don't decrease at all after a convolutional layer. So, the number of the layers can be unlimited.

However, the more layers in the network, the more unstable is its behavior with random weights, *i.e.* training from random initial state doesn't converge. For that reason, several networks I tried didn't train properly. The file contains one network that did converge to a solution. 

### Results

The "more deeper network" fitted better the training set, but the validation accuracy was similar or worse than of the "best network" (network 1 from file 5).

Probably, techniques that help combat overfitting (regularization and normalization) are important for improving test accuracy further.

## Fine tuning ([file 7](HW2_file7.ipynb), [file 7b](HW2_file7b.ipynb), and  [file 7bRMS](HW2_file7bRMS.ipynb))

Before doing K-fold validation and final training, I fine-tuned the parameters of the training of the "best network", mainly the settings for data augmentation

### Data augmentation settings from the class ([file 7](HW2_file7.ipynb))

Since I used other setting for data augmentation (not the settings from the example presented in the class), I checked the behavior of the settings from the class on my model.

For the "best model", these settings resulted in a worse validation accuracy (~82% instead of ~88%). 

Because the training accuracy was also low (78%), the more complicated model could possibly attain better results. Indeed, the "more deeper network" (file 6) reached 88% validation accuracy and 94% training accuracy. Still, this validation accuracy was not better than the the best validation accuracy attained  with other settings by the simpler "best network" (network 1 from file 5).

### Intermediate  data augmentation settings ([file 7b](HW2_file7b.ipynb))

Since the data augmentation settings from file 5 resulted in overfitting for the "best model", I tweaked them a little toward the settings from file 7. As the result, the overfitting of the "best model" was reduced. Even though the validation accuracy increased only slightly, if any, the new data augmentation setting are subjectively a little better. 

This concludes the fine tuning of the data augmentation, although the investigation was not complete and further fine tuning might be beneficial.

### RMSprop optimizer ([file 7bRMS](HW2_file7bRMS.ipynb))

Performance of SGD optimizer (used everywhere else in the project) was compared to the RMSprop optimizer. Although not exactly clear why, fitting of the "best model" using RMSprop optimizer resulted in significantly worse prediction than using SGD optimizer.

## K-fold validation ([file 8](HW2_file8.ipynb))

K-fold validation (with K=5) was performed on the "best model" (file 7b).

### Notes on implementation

Because the training time is very large, in order to be able to resume calculations, the training is done in iterations of 100 epochs. The models are saved to Google Drive between the iterations. 

Among the 5 sets of training/validation data used for K-fold validation, the last set was identical to the set used in all previous studies.

### Results

After 300 epochs, the validation accuracy was 88.00% (+/- 0.26% STD). However, because the learning rate was reduced stochastically, the training process was at different level of completion for different sets of data. Therefore, the training was performed for another 100 epochs. After total 400 epochs the validation accuracy was 88.40% (+/- 0.20% STD).

Quite noticeably, the last set of data (the set that was used in all previous studies) produced the best validation accuracy among all sets both after 300 epochs and after 400 epochs, which indicated that our selection of the networks could be bias toward this set.

## Final training on full training set ([file 9](HW2_file9.ipynb))

The "best network" was selected to be trained on the full training set and evaluated on the test set.

All the parameters were the same as in file 8, including 400 epochs for training.

### Notes on implementation

The code is similar to file 8.

### Results

The final validation accuracy was 89%, the training accuracy was 91%. These values were reached after 300 epochs of training and basically stayed unchanged for the last 100 epochs.
