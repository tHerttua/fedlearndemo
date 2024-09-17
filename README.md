# Description

This is a demo of a federated learning pipeline, where there is a server side global model and five client side models. The clients train on their own local datasets and send the model updates back to the server for federated averaging.

For this task, Tensorflow was chosen and it is implemented in Python, because of  their popularity and effectiveness, as well as my knowledge with these tools.

The dataset is CIFAR-10, and it simulates a non Independent and Identically Distributed data, as it is more interesting to work with, and often in real implementations there is no luxury of IID data.
In short, each of the clients have a slight variance in their data sets.

To simulate the behaviour and federated learning (FL) principles, the approach borrows from Agent Based Modeling, which is apt as the clients and the server can be thought of as individual agents that interact with each other in the environment indirectly through the server agent. This approach also allows simulation of different scenarios, such as considering security.

The demonstration script gathers loss and accuracy over the communication rounds and displays how the clients and the global model performance evolves over the rounds.

The main focus was to create simple simulation with limited time, which turned out great. However, some details and additional things as well as hyperparameter search was left out for this version.
As this is just a demonstration, the logging is crumbled within the script that acts as the entrypoint (main.py) instead of proper logging module, and no extensive testing is done aside from a few integration tests.
To keep this script hardware agnostic and simple enough, the federated learning is done in a loop with unique clients instead of creating, for example, docker containerization, or another more sophisticated way of simulation.
The result it provides is nondeterministic due to how I implemented the non-IID data, but without further parameter searching the resulting global model test accuracy is roughly 60% with these hyperparameters.

test folder includes two different integration tests to verify the agents indeed are unique, and another to understand the distribution of data between the clients.

## handle_data.py

Takes care of the data by loading and stratifying it. In order to simulate different scenarios, a custom data partioning function was made. The idea is to simulate a scenario, where there are n amount of clients that deal with similar data, but each of them are interested in different, albeit overlapping, classification targets.
As the data is randomly partitioned, it may cause one or more classes to be more underpresented. This design choice was made for simulation purposes and to add more variance in the results. It is still desgined not to share any raw data between the different agents of the system. 

## agents.py 

This module defines the two different type of agents, namely Client and Server, and it can be expanded for different type of agents for simulating different scenarios.I had a lot of problems with trying to set up TensofFlow Federated Learning module, which is a shame, so instead I implemented a type of federated averaging with numpy instead of using this TFF frameworks prebuilt functions.

To demonstrate differential privacy, simple methods are implemented:
Weight clipping limits how effective any one of the clients data is by clipping updates to a predefined norm.
Noise addition can be controlled by 'privacy budget' and the aim is to obscure if some certain data was used in training.
Note that optimal values for these differential privacy techniques were not tested; it is always a trade-off between utility and privacy with these simple methods.

Aside from searching for the most optimal hyperparameters considering the model architecture and learning rate, etc., Tensorflows Keras tuner could be possibly used in order to find the most suitable parameters for the differential privacy as well.

## CNN_model

A very simple neural network with CNN architecture is implemented as the model for this task. This model's architecture is the same across all parties (clients and the server).
Convolutional layers with convolutional filters to extract features.
Pooling layers downsamples the feature maps to reduce computational cost
Flatten layer converts the 2D map into a vector
ReLU activation is efficient and is placed to prevent vanishing gradient.
Dense layers process the features into 10 different classes (as per Cifar-10)
Softmax normalizes the output probabilities to sum up to 1 between the classes

optimizer, loss and metrics can be passed, but the preset ones are:
Adam as the optimizer as it is popular and results in fast convergence.
sparse_categorical_crossentropy because it is suitable for multiclass classification.
Accuracy as the default metrics to track how often the prediction is correct.

A regularizer helps to prevent overfitting, and as the model is very simple, instead of Dropout layer an L2 regularizer is added to penalize large weights.

In this particular dataset it is not critical to monitor for example false positives
too closely, so following the accuracy metric should be enough.

## main.py

Essentially acts as the environment for the agents to interact in by simulating the federated learning process and printing out information into the console about the progress so far. Uses all the modules and strings them together to run a simulation.

Step by step description:
Loads the data, normalizes the data and splits it into training and testing datasets.
Using the chosen parameters for number of clients, communication rounds, number of epochs and differential privacy parameters, initializes the Server and the Client agents. Here each of the agents own datasets are assigned to them before moving to simulating the communication rounds.

After everything is initialized, the training loop begins:
1. Global model is initialized in Server with random weights.
2. Each of the Clients local models are initialized with the global model's weights.
3. Each of the Clients local models are trained with their assigned data for the number of epochs. If Differential Privacy parameters are set, their effects are added to the weights.
4. Each of the local models weights are appended to a list.
5. The list is passed to the Server agent, which performs weight aggregation and sets the global model weights as the result of the aggregation.
6. Evaluation metrics are stored in their respective lists
7. New training loop (or communication round) may begin

After the described training loop is ran for the number of set communication rounds, the gathered results are plotted with matplotlib.


# How to run

The program is made on a Linux system and it has not been tested on a windows or macOS.
It is advised to create a virtual environment.

Create virtual environment:
python -m venv .venv

Install modules:
python -m pip install -r requirements.txt

Run the main.py script using a debugger or just cmd:
python main.py

The main hyperparameters can be changed in main.py 

# Security and risk discussion

There are different ways to enhance the security in federated learning. Potential risks are the server getting compromised, which would lead to the adversarial part gaining access to sensitive data for example. Another risk is that there is a malicious, or otherwise adversarial client among the good clients. It could be either defect or on purpose try and poison the global model and this way indirectly attacking the other clients as well. We shall very shortly discuss these scenarios.

In this demo, a very simple noise addition and weight clipping was shown. Another simple measure to implement would have been secure aggregation with either 'additive secret sharing' where the clients split their updates into random parts and sends them to the server, which can construct the update from these shared parts. Another fairly simple method could have been 'homomorphic encryption', which works by the clients encrypting their updates before sending them to the server, which in turn aggregates the updates using operations such as summing these encrypted updates. 
There are more sophisticated methods and for example tensforflow privacy offers premade solutions for tensorflow.

Also, the communication overhead and computational cost should be taken into account, especially when working with less capable devices for example certain IoT systems. For example 'additive secret sharing' is less computationally heavy, and only sending the difference between initial weights and updated weights could be a clever way to reduce the use of system resources. 

Protecting the system against malicious clients could be trickier. Some interesting ideas come from the use of Siamese networks or Game theoretic approaches, with the very core of the idea lying in Anomaly Detection: observing anomalous patterns in model updates. Siamese network compare incoming client updates to a reference dataset, and these networks can effectively flag suspicious patterns. The core idea is to train a Siamese network to learn representations that distinguish between legitimate and malicious updates. This can be achieved by using a contrastive loss function that encourages the network to produce similar embeddings for normal updates and dissimilar embeddings for anomalous ones. After each communication round, the server could use a Siamese network to compare the updates from client with a reference updates. 

Game theory could provide another interesting framework for understanding and countering malicious client behavior. By modeling the interaction between the server and malicious clients as a game, we can analyze the incentives and strategies of both parties. Zero-sum games, where one player's gain is another's loss, can be used to model adversarial relationships. Evolving games, on the other hand, allow for dynamic adaptation as malicious clients change their attack strategies. Cooperative games, and for example, counting the Shapley value of clients based on how good their updated contribute to the overall metrics could be useful, but it becomes computitionally heavy with many clients when having to consider all the different coalitions.
