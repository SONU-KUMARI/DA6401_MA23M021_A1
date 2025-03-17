import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from ma23m021_assignment1_ import sgd, trainn, adam, nesterov, momentum, nadam, rmsprop
import wandb


def main():
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('--wandb_project', type=str, required=True, help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('--wandb_entity', type=str, required=True, help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    
    parser.add_argument('-sid', '--wandb_sweepid', type=str, default=None, help='Wandb Sweep Id to log in sweep runs the Weights & Biases dashboard.')
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=["mnist", "fashion_mnist"], help='Dataset choices: ["mnist", "fashion_mnist"]')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs to train neural network.')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size used to train neural network.')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=["mean_squared_error", "cross_entropy"], help='Loss function choices: ["mean_squared_error", "cross_entropy"]')
    parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=["stochastic", "momentum", "nag", "rmsprop", "adam", "nadam"], help='Optimizer choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate used to optimize model parameters')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum used by momentum and nag optimizers.')
    parser.add_argument('-beta', '--beta', type=float, default=0.99, help='Beta used by rmsprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999, help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-08, help='Epsilon used by optimizers.')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005, help='Weight decay used by optimizers.')
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=["random", "xavier"], help='Weight initialization choices: ["random", "xavier"]')
    parser.add_argument('-nhl', '--num_layers', type=int, default=5, help='Number of hidden layers used in feedforward neural network.')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128, help='Number of hidden neurons in a feedforward layer.')
    parser.add_argument('-a', '--activation', type=str, default='tanh', choices=["identity", "sigmoid", "tanh", "ReLU"], help='Activation function choices: ["identity", "sigmoid", "tanh", "ReLU"]')
    
    args = parser.parse_args()
    
    # Initialize WandB
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    
    def one_hot_encoding(y, num_classes=10):
        return np.eye(num_classes)[y]



    # Load dataset
    dataset_dict = {"mnist": mnist.load_data, "fashion_mnist": fashion_mnist.load_data}
    (X_train, y_train), (X_test, y_test) = dataset_dict[args.dataset]()
    X_train, X_test = X_train.astype(np.float32) / 255.0, X_test.astype(np.float32) / 255.0
    X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    

    y_train = one_hot_encoding(y_train)
    y_valid = one_hot_encoding(y_valid)
    y_test = one_hot_encoding(y_test)


    # Select optimizer
    optimizer_dict = {'sgd': sgd, 'adam': adam, 'momentum': momentum, 'nesterov': nesterov, 'rmsprop': rmsprop, 'nadam': nadam}
    optimizer_func = optimizer_dict[args.optimizer]
    
    # Train model
    weights, biases = trainn(X_train, y_train, X_test, y_test, num_hidden_layers=args.num_layers, num_neurons=args.hidden_size, 
                             num_epochs=args.epochs, batch_size=args.batch_size, learningg_rate_=args.learning_rate, 
                             optimizer_func=optimizer_func, init_method=args.weight_init, activation_func=args.activation)

if __name__ == '__main__':
    main()