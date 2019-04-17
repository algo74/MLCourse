# Solution to HW3

*by Alexander Goponenko*

## [Problem 1](HW3_Problem1.ipynb)

I used more aggressive data augmentation (than in the notebook shown in the class) and dropouts for fully connected network to get better results.

Also I used `preprocess_input` from `keras.applications.vgg19` in the `ImageDataGenerator`. It improved accuracy considerably.

Having a more complicated "head" seemed to be unjustified given just one binary classifier and the tendency of overfitting. I tried to add another convolutional layer, but it did not do any good.

*Final validation accuracy: 97%*

## Problem 2
### [Fine-tuning the last convolutional layer](HW3_Problem2.ipynb)

Unfreezing only the last convolutional layer of VGG19 produced mediocre improvement if any. It is interesting to see what will happen if the whole network is allowed to train.

*Validation accuracy: 97%*

### [Fine-tuning the whole VGG19](HW3_Problem2_fulltraining.ipynb)

Unfreezing the whole network improved the validation accuracy by another percent, which is better than unfreezing only the last convolutional layer.

Some overfitting was observed. Possibly more aggressive data augmentation and/or more dropouts could help reach even better results (but the training set is quite small).

*Final validation accuracy: 98%*

## [Problem 3](HW3_Problem3.ipynb)

Unfortunately, `K.gradients` did not work with the original model. I had to rebuild the model and then set its weights from the old model (otherwise I would have to redo Problems 1 & 2).

I visualized one properly classified dog, one properly classified cat, one cat missclassified as dog, and one dog missclassified as cat. 

Areas that resemble cats more than dogs are blue and areas that resemble dogs more than cats are red/yellow.

## [Problem 4](HW3_Problem4.ipynb)

I had to rebuild the model for similar reasons as in Problem 3. 

I visualized four classes: 'true cat', 'true dog', 'fake cat', and 'fake dog' and showed results for two values of perplexity. 

The convolutional layers did quite decent job separating dogs from cats. The "head" of the NN was able to find a natural border between the classes and also propery classify some outsiders. It might be overly aggressive in assuming cats as "outsider" dogs.
