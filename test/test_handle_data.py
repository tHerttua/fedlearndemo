import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../.'))
from handle_data import load_and_preprocess_data, partition_data

def analyze_data_distributions():

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    client_data = partition_data(x_train, y_train, num_clients=5)

    client_data_summary = {}
    for client_id, (client_x, client_y) in client_data.items():
        client_data_summary[client_id] = {}
        for class_label in np.unique(client_y):
            count = np.count_nonzero(client_y == class_label)
            total_samples = len(client_y)
            percentage = (count / total_samples) * 100
            client_data_summary[client_id][class_label] = percentage

    for client_id, category_percentages in client_data_summary.items():
        print(f"\nClient ID: {client_id}")
        print("-" * 30)
        for category, percentage in category_percentages.items():
            print(f"Category {category}: {percentage:.2f}%")


