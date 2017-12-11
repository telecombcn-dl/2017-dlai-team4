# Dog Breed Identification
## Kaggle Dog Breed Identification Challenge

Welcome to our GitHub page! You will find here a review of our work.

Our team is composed by:

![Mattia image](https://github.com/auderaza/test/blob/master/mattia.png)|![Aude image](https://github.com/auderaza/test/blob/master/aude.png)|![Paolo image](https://github.com/auderaza/test/blob/master/paolo.png)            
------------ | ------------ |------------ 
  Mattia LECCI |   Aude RAZAFIMBELO |   Paolo TESTOLINA

The **objective** of our work is to determine the bread of a dog from an image. To do so, we perform the different following tasks.


## Architecture

To launch our work, we use an already existing code ([MNIST starter code](https://github.com/yashk2810/MNIST-Keras/blob/master/Notebook/MNIST_keras_CNN-99.55%25.ipynb)). It permits us to have a first approach on manipulating a neural network. The data processed are the MNIST data.

The CNN is composed of the following layers: 

* **Convolution** layer
* **Activation** layer
* **Pooling** layer

Then after repeating it a few times, the CNN is **Fully Connected** in order to classify the samples. Finally a **Batch Normalization** is applied.

## Training

Then from this, we apply this algorithm to the dogs samples. This experiment provides us great results!

*RESULTS*

## Transfer learning

Secondly, we use another tool ([Keras VGG19 starter](https://www.kaggle.com/orangutan/keras-vgg19-starter/notebook)). The data processed are the dogs samples. The main steps of the code:
* the breeds are one-hot encoded for the final submission 
* there are in total 120 different breeds/ classes
* the images are resized
* the test, train and validation sets are defined
* the CNN architecture is built using a **pre-trained model VGG19** and adding a new top layer

From here, we **experiment with the code** by trying to apply different hyperparameters, layers, loss and activation functions.
The idea is to use the pre-trained VGG19 model.

![VGG19 image](https://github.com/telecombcn-dl/2017-dlai-team4/blob/master/images/vgg19.png)  

## Optimization

 We then **otpimize** the current algorithm by through different ways.

1. VGG19 optimization
*RESULTS* - Overfitting of the model

2. Add hyperparameters for optimization: **Adam optimizer**

3. Add hyperparameters for optimization: **Sgd optimizer**

4. Combination of VGG19 and VGG16
*RESULTS*

5. Add BatchNormalization to VGG16: **Xception optimization**

## All in all

Through this project, we were able to manipulate a bunch of new technologies such as Python, TensorBoard and GCloud. Moreover, we learn how to design and improve basic neural networks. 
