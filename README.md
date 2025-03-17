# DA6401 Deep Learning Assignment--1

# Classification of Fashion-MNIST dataset using Neural Network:

This repository contains a complete implementation of a neural network model designed to classify images from the Fashion-MNIST dataset. The project supports various optimization algorithms like SGD, Nesterov, Adam etc. and hyperparameter tuning using Weights & Biases (wandb) sweeps. This project is built from scratch using numpy, pandas, matplotlib and dataset loaded from keras.
And the finest configuration of hyperparameters was also run on MNIST dataset. 

## Dataset Overview:

The Fashion-MNIST dataset consists of 60,000 training images and 10,000 testing images. It contains 10 classes (e.g., T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot).

Each image is 28x28 pixels in grayscale. To prepare the data:
 Flattened each image into a vector of size 784.
 Normalized pixel values to the range [0,1].
 One-hot encoded labels into vectors of length 10.



## Model Architecture:

The network supports multiple hidden layers, activation functions, and weight initialization methods. Each experiment runs with different hyperparameters and optimizers.

### Neural Network Initialization:

The model is initialized with:
  Input size: 784 (28x28 images)
  Output size: 10 (one neuron per class)
  Hidden layers: Configurable (3, 4, or 5 layers)
  Number of neurons per layer: Configurable (32, 64, 128)
  Weight initialization: Xavier or random

The initialization is handled by the param_init function which is flexible to take hidden layers, neurons per hidden layer and which returns weight and bias after initialization.


## Forward and Backward Propagation:

### Forward Propagation
The hidden layer in neural network uses a selected activation function (ReLU, Sigmoid or Tanh), and the output layer applies Softmax to yield class probabilities. So we initialized the parameters and feeded to forward propagation and yielded the probabilities for the classes. Also, calculated the loss for each epoch using gradient descent.

### Backward Propagation

The model computes Cross-Entropy Loss and Mean Squared Error (MSE), two widely used loss functions for classification problems. In Cross-Entropy, we measures how well the predicted probability distribution aligns with the true labels, while in MSE, we calculate the squared difference between the predicted and actual outputs.

The gradients of the loss with respect to each weight and bias are computed using backward flow. These gradients indicate how much each parameter contributed to the overall error. The weights and biases are then updated in the opposite direction of the gradients — effectively reducing the loss — according to the selected optimizer’s specific update rules (e.g., momentum for faster convergence or adaptive learning rates in Adam and RMSprop). This iterative process continues for each batch and across multiple epochs, gradually improving the model’s performance.

## Optimizers:
- **Stochastic Gradient Descent (SGD)**  
- **Momentum-based Gradient Descent (MGD)**  
- **Nesterov Accelerated Gradient (NAG)**  
- **RMSprop**  
- **Adam**  
- **Nadam**  



## Model Training:

The function training_model() handles training with different hyperparameter combinations provided by the wandb sweep configuration.

### Hyperparameters

 ### Hyperparameters for the Sweep

- **Learning rate:**  
  - 0.001  
  - 0.0001  

- **Hidden layers:**  
  - 3  
  - 4  
  - 5  

- **Nodes per layer:**  
  - 32  
  - 64  
  - 128  

- **Activation functions:**  
  - Sigmoid  
  - ReLU  
  - Tanh  

- **Optimizers:**  
  - SGD  
  - Momentum  
  - Nesterov  
  - RMSprop  
  - Adam  
  - Nadam  

- **Batch size:**  
  - 16  
  - 32  
  - 64  

- **Epochs:**  
  - 5  
  - 10  

- **Weight initialization:**  
  - Xavier  
  - Random  


Best configuration is determined based on Validation Accuracy.

## Hyperparameter Sweep Configuration:

Weights & Biases Sweep Setup
The sweep explores different configurations using Grid Search and Bayesian Optimization methods and metric is used to maximize accuracy or to minimize the loss.
I ran these configurations for reasonable number of counts. And plots for these metrics is also visualized


Cross-Entropy vs. MSE Sweeps
To compare performance between Cross-Entropy and MSE loss functions, results are logged in two separate Weights & Biases projects:
- **Cross-Entropy Loss Project:** MA23M021_A1_Q8_CROSS  
- **MSE Loss Project:** MA23M021_A1_Q8_MSE  



## Confusion Matrix :
A confusion matrix is also plotted (plots true vs. predicted labels) on the best configuration of hyperparameters from the sweep in WANDB that is in question 7 of the assignment.

## MNIST Dataset Prediction:
The model is further tested on the original MNIST dataset with three different top-performing hyperparameter configurations which we extracted from fashion MNIST dataset from experimentation.




Weights & Biases Project Report link:

https://wandb.ai/ma23m021-iit-madras/MA23M021_A1/reports/MA23M021_Assignment-1--VmlldzoxMTcwODcwMg?accessToken=6y55e1hswd5y1v9lwv4vwl6fa54vo31vvosj5o8go1f88c1zcf87d13siklzjo5a



# How to Run Code:

I have given ma23m021_Assignment1.py and train.py files. These files have to be in same directory after downloading as I am importing the functions from this file in train.py file. And also add your API key in the ma23m021_Assignment1.py file in line number 19 to see the run in your wandb.
