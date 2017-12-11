# Dog Breed Identification
## Kaggle Dog Breed Identification Challenge

Welcome to our GitHub page! You will find here a review of our work.

Our team is composed by:

![Mattia] (https://github.com/telecombcn-dl/2017-dlai-team4/blob/master/images/mattia.png)|![Aude] (https://github.com/telecombcn-dl/2017-dlai-team4/blob/master/images/aude.png)|![Paolo] (https://github.com/telecombcn-dl/2017-dlai-team4/blob/master/images/paolo.png)            
------------ | ------------ |------------ 
  Mattia LECCI |   Aude RAZAFIMBELO |   Paolo TESTOLINA

The **objective** of our work is to determine the bread of a dog from an image. To do so,we perform the different following tasks.


## Task 1

To launch our work, we use an already existing code ([Keras VGG19 starter](https://www.kaggle.com/orangutan/keras-vgg19-starter/notebook)). The data processed are the CSV files which contain the labels' breeds and the samples of the animals to be submit. The main steps of the code:
* the breeds are one-hot encoded for the final submission; there are in total 120 different breeds/ classes
* the images are resized
* the test, train and validation sets are defined
* the CNN architecture is built using a **pre-trained model VGG19** and adding a new top layer

*This model provides a very low accuracy as the pre-trained weights are not used.*

*RESULTS*

From here, we **experiment with the code** by trying to apply different hyperparameters, layers, loss and activation functions.
The idea is to use the pre-trained VGG19 model and do fine-tuning.

![VGG19] (https://github.com/telecombcn-dl/2017-dlai-team4/blob/master/images/vgg19.png)  


## Task 2

Our second objectif is to **otpimize** the current algorithm. + fine tuning

1. VGG19 optimization
We use predetermined optimizer. Plus, we pply data augmentation and fitting.

*RESULTS*

2. Add hyperparameters for optimization: **Adam optimizer**

*RESULTS - VGG19 optimization adam*

3. Add hyperparameters for optimization: **Sgd optimizer**

*RESULTS - not that good*

4. Combination of VGG19 and VGG16

*REULTS of the perfromance and overall reults evaluation*

5. Add BatchNormalization to VGG16: **Xception optimization**

*RESULTS - quite good*


## Task 3

To go further, we build another CNN, based on the previsous VGG19 model, which perform **Multiclassification**.
The idea is to add a Convolution layer, an Activation layer and a Pooling layer. Then after repeating it a few times, the CNN is Fully Connected in order to classify the samples. Finally a Batch Normalization is applied.

*RESULTS*

## All in all

Through this project, we learn a lot about neural networks and especially how to design and improve them. 
