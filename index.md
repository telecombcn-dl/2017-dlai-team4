# Welcome to our GitHub page! You will find here a review of our work.

Our team is composed by:

| ![Mattia-image](https://github.com/telecombcn-dl/2017-dlai-team4/blob/master/images/mattia.png)| ![Aude-image](https://github.com/telecombcn-dl/2017-dlai-team4/blob/master/images/aude.png) | ![Paolo-image](https://github.com/telecombcn-dl/2017-dlai-team4/blob/master/images/paolo.png) |        
| :---: | :---: | :---: | 
| Mattia LECCI | Aude RAZAFIMBELO | Paolo TESTOLINA |

The **goal** of our work is to classify dogs images according to their breeds. To do so, we perform the different tasks presented here.


## Architecture

To launch our work, we use an already existing code ([MNIST starter code](https://github.com/yashk2810/MNIST-Keras/blob/master/Notebook/MNIST_keras_CNN-99.55%25.ipynb)). It permits us to have a first approach on manipulating a neural network. 

The CNN is composed by the following layers: 

* **Convolution** layer
* **Activation** layer
* **Pooling** layer

Then after repeating it a few times, the CNN is **Fully Connected** in order to classify the samples. Finally a **Batch Normalization** is applied.

## Training

From this, we apply this algorithm to the dogs samples. This experimentation provides us great results: the accurary reaches 0.22.

![MNIST best model result - image](https://github.com/telecombcn-dl/2017-dlai-team4/blob/master/images/mnist.png)

## Transfer learning

Secondly, we use another tool ([Keras VGG19 starter](https://www.kaggle.com/orangutan/keras-vgg19-starter/notebook)). The data processed are still the dogs samples. The main steps of the code:

* the breeds are one-hot encoded for the final submission 
* there are in total 120 different breeds/ classes
* the images are resized
* the test, train and validation sets are defined
* the CNN architecture is built using a **pre-trained model VGG19** and adding a new top layer

From here, we **experiment with the code** by trying to apply different hyperparameters, layers, loss and activation functions.
The idea is to use the pre-trained VGG19 model.

![VGG19 model - image](https://github.com/telecombcn-dl/2017-dlai-team4/blob/master/images/vgg19.png)  

## Optimization

 We then **otpimize** the current algorithm by through different ways.

1. VGG19 optimization

![VGG19 results - image](https://github.com/telecombcn-dl/2017-dlai-team4/blob/master/images/vgg19result.png)

We can see that the model is overfitting: the final value of the accuracy almost reach 1.00 and the final value of the loss almost rech 0.00. 

2. Add hyperparameters for optimization: **Adam optimizer**

3. Add hyperparameters for optimization: **Sgd optimizer**

4. Combination of VGG19 and VGG16

![VGG16 results - image](https://github.com/telecombcn-dl/2017-dlai-team4/blob/master/images/vgg16.png)

The accuracy reaches 0.33.

5. Add BatchNormalization to VGG16: **Xception optimization**

## What we learned from the project

Through this project, we were able to manipulate a bunch of new technologies such as:
* Python
* TensorBoard
* GCloud
* GitHub.
Moreover, we learned how to design a basic neural networks. We were able to manipulate the given network in order to improve it and fit it to our model.
As we were beginners in pratical neural network, we can now assume we have gained some basics!

## More information

Here are the [Slides of our presentation](https://docs.google.com/presentation/d/1Ll6pUaIbTFKg-3NNc8YemHoBIV9hcGibhmGtIceK0Rc/edit?usp=sharing), that provide further information.
