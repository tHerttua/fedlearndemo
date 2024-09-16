import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../.'))
import agents
import handle_data
import CNN_model

def test_agent_uniqueness():
    (x_train, y_train), (x_test, y_test) = handle_data.load_and_preprocess_data()
    client_data = handle_data.partition_data(x_train, y_train, num_clients=5, num_shards=10, focus_ratio=0.6)
    clients = [agents.Client(data, CNN_model.create_model(), verbose=0) for data in client_data.values()]

    # Test if each client has a unique model instance
    model_ids = [id(client.model) for client in clients]
    assert len(model_ids) == len(set(model_ids)), "Each client should have a unique model instance"

    # Print model IDs for debugging purposes
    for i, model_id in enumerate(model_ids):
        print(f"Client {i} model id: {model_id}")
