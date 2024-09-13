import tensorflow as tf
import numpy as np
from CNN_model import create_model
from handle_data import partition_data, load_and_preprocess_data
from agents import Server, Client

VERBOSE=1

def simulate_federated_learning(x_train, y_train, x_test, y_test, num_clients=5, num_rounds=5, num_epochs_local=2):
    # Initialize global model
    global_model = create_model()
    server = Server(global_model)
    
    # Partition the training data between clients and create models for each client
    client_data = partition_data(x_train, y_train, num_clients=num_clients, num_shards=10, focus_ratio=0.8)
    clients = [Client(data, create_model(), verbose=VERBOSE) for data in client_data.values()]


    # Make sure that each of the clients are unique
    print("Initialized models")
    for i, client in enumerate(clients):
        print(f"Client {i} model id: {id(client.model)}")

    # Simulate communication rounds
    for round_num in range(num_rounds):
        print(f"Round {round_num+1}/{num_rounds}")
        
        client_weights = []
        
        # Train each client and collect weights by looping over them
        for client in clients:
            print(f"Training model id: {id(client.model)}")

            # Share the global model's latest weights with all clients
            client.model.set_weights(global_model.get_weights())

            # Local training on an individual client
            weights = client.train(epochs=num_epochs_local)
            client_weights.append(weights)
        
        # Aggregate client updates to form a new global model
        server.aggregate(client_weights)
        
        # Evaluate the global model
        test_loss, test_accuracy = server.evaluate(x_test, y_test)
        print(f"Test Loss on global model: {test_loss}, Test Accuracy: {test_accuracy}")


if __name__ == "__main__": 

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    simulate_federated_learning(x_train, y_train, x_test, y_test, num_clients=5, num_rounds=2, num_epochs_local=2)



