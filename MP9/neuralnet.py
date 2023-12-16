# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP9. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        h = 128

        self.fc1 = nn.Linear(in_features = in_size, out_features = h)
        self.fc2 = nn.Linear(in_features = h, out_features = out_size)
        self.optimizer = optim.SGD(self.parameters(), lr = self.lrate)
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
    
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        self.optimizer.zero_grad()
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # Normalize training
    train_set = (train_set - torch.mean(train_set, dim=0)) / torch.std(train_set, dim=0)
    
    # Variables and functions
    lrate = 0.01
    loss_fn = nn.CrossEntropyLoss()
    input_size = train_set.shape[1]
    output_size = len(np.unique(train_labels))
    
    # Create the neural network.
    net = NeuralNet(lrate=lrate, loss_fn=loss_fn, in_size=input_size, out_size=output_size)
    
    # Retrieve the corresponding labels inside a dataset
    train_data = get_dataset_from_arrays(train_set, train_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        
    # Training data loop
    losses = np.zeros(epochs)
    for i in range(epochs):
        for batch in train_loader:
            losses[i] += net.step(batch['features'], batch['labels'])

    yhats = np.argmax(net.forward(dev_set).detach().cpu().numpy(), axis=1).astype(int)

    return losses.tolist(), yhats, net
