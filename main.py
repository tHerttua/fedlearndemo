import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from CNN_model import create_model
from handle_data import partition_data, load_and_preprocess_data
from agents import Server, Client

VERBOSE=1

def simulate_federated_learning(x_train, y_train, x_test, y_test, num_clients=5, num_rounds=5, num_epochs_local=2, clip_norm=None, privacy_budget=None):
    # Initialize global model
    global_model = create_model()
    server = Server(global_model)
    
    # Partition the training data between clients and create models for each client
    client_data = partition_data(x_train, y_train, num_clients=num_clients, num_shards=10, focus_ratio=0.8)
    clients = [Client(data, create_model(), verbose=VERBOSE, clip_norm=clip_norm, privacy_budget=privacy_budget) for data in client_data.values()]

    # Initialize histories for plotting
    client_loss_history = {i: [] for i in range(num_clients)}
    client_accuracy_history = {i: [] for i in range(num_clients)}
    server_loss_history = []
    server_accuracy_history = []

    print("Initialized models")
    for i, client in enumerate(clients):
        print(f"Client {i} model id: {id(client.model)}")

    # Simulate communication rounds
    for round_num in range(num_rounds):
        
        client_weights = []
        
        # Train each client and collect weights by looping over them
        for i, client in enumerate(clients):
            print(f"Round {round_num+1}/{num_rounds}")
            print(f"Training model id: {id(client.model)}")

            # Share the global model's latest weights with all clients
            client.model.set_weights(global_model.get_weights())

            # Local training on an individual client
            weights = client.train(epochs=num_epochs_local)
            client_weights.append(weights)

            # Evaluate each client's model on their local data
            loss, accuracy = client.model.evaluate(client.data, client.labels, verbose=0)
            client_loss_history[i].append(loss)
            client_accuracy_history[i].append(accuracy)
        
        # Aggregate client updates to form a new global model
        server.aggregate(client_weights)
        
         # Evaluate the global model
        test_loss, test_accuracy = server.evaluate(x_test, y_test)
        server_loss_history.append(test_loss)
        server_accuracy_history.append(test_accuracy)
        print(f"Test Loss on global model: {test_loss}, Test Accuracy: {test_accuracy}")


    # Plot server performance
    plt.figure()
    plt.plot(server_loss_history, label="Global Model Test Loss")
    plt.plot(server_accuracy_history, label="Global Model Test Accuracy")
    plt.title("Global Model Performance Over Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Loss / Accuracy")
    plt.legend()
    plt.show()

    # Plot client metrics
    plt.figure()
    for i in range(num_clients):
        plt.plot(client_accuracy_history[i], label=f"Client {i} Accuracy")
        plt.plot(client_loss_history[i], label=f"Client {i} Loss", linestyle='--')
    plt.title("Client Metrics Over Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy / Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__": 

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    simulate_federated_learning(x_train, y_train, x_test, y_test, 
                                num_clients=5, 
                                num_rounds=10, 
                                num_epochs_local=3, 
                                clip_norm=None, 
                                privacy_budget=0.01)

