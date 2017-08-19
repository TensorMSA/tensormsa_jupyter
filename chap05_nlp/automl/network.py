"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score

class Network():
    """Represent a network and let us operate on it.
    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.
        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.
        Args:
            network (dict): The network parameters
        """
        self.network = network

    def train(self, dataset, skip_real_train = False):
        """Train the network and record the accuracy.
        Args:
            dataset (str): Name of dataset to use.
        """
        if self.accuracy == 0.:
            if(skip_real_train == False) : 
                self.accuracy = train_and_score(self.network, dataset)
            else : 
                self.accuracy = random.randrange(0,100)/100
            print("Test condition of each network : {0}".format(self.network))
            print("Test result of each network : {0}".format(self.accuracy))

    def print_network(self):
        """Print out a network."""
        print('-'*80)
        print("Top 5 result") 
        print(self.network)
        print("Network accuracy: %.2f%%" % (self.accuracy * 100))
        print('-'*80)