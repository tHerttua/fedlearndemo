# Description and Discussion

This is a demo of a federated learning pipeline, where there is a server side global model and five cilent side models. The clients train on their own local datasets and send the model updates back to the server for federated averaging.

For this task, tensorflow was chosen and it is implemented in Python, because of  their popularity and effectiveness, as well as my knowledge with these tools.

The dataset is CIFAR-10, and it simulates a non Independent and Identically Distributed data, as it is more interesting to work with, and often in real implementations there is no luxury of IID data. 
In short, each of the clients have a slight variance in their data sets.

To simulate the behaviour and federated learning (FL) principles, the approach is based on Agent Based Modeling, which is apt as the clients and the server can be thought of as individual agents that interact with each other in the environment indirectly through the server agent. This approach also allows simulation of different scenarios considering security.

As this is just a demonstration, the logging is crumbled within the script that acts as the entrypoint (main.py) instead of proper logging module, and no extensive testing is done aside from a few integration tests.
To keep this script hardware agnostic and simple enough, the federated learning is done in a loop with unique clients instead of creating, for example, docker containerization.

## handle_data.py

Takes care of the data by loading and partitioning it. In order to simulate different scenarios, a custom data partioning function was made. The idea is to simulate a scenario, where there are n amount of clients that deal with similar data, but each of them are interested in different, albeit overlapping, classification targets.

## agents.py 

This module defines the two different type of agents and it can be expanded for different type of agents for simulating different adversarial attacks.
To achieve 

## CNN_model

A very simple neural network with CNN architecture is implemented as the model for this task. This model's architecture is the same across all parties (clients and the server). Convolutional layers extract the features, pooling layers reduce the dimensionality of the feature maps, and fully connected layers classify the features.

Activation function
Optimizer
Loss function

## main.py

Essentially acts as the environment for the agents to interact in by simulating the federated learning process and printing out information into the console about the progress so far.


# How to run

# Security and risk mitigation

## Siamese networks

## Game theoretic approaches