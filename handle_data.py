import tensorflow as tf
import numpy as np

def load_and_preprocess_data():
    """
    Loads the CIFAR-10 dataset
    Normalizes the pixel values between 0 and 1
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize 
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)


def partition_data(x_train, y_train, num_clients=5, num_shards=10, focus_ratio=0.6):
    """
    Partition the dataset in a non-IID fashion where each client will gravitate towards two
    specific classes, but will also have small amounts of data from all classes.
    
    The idea is to simulate a scenario, where there are n amount of clients that deal with similar data,
    but each of them are interested in different, albeit overlapping, classification targets.
    
    Uses Dirichlet distribution to create a non-uniform probability distribution over the non-focus
    classes.
    
    num_shards: Number of data shards to divide the dataset into.
    focus_ratio: The proportion of data that should focus on two specific classes for each client.

    Returns a dictionary where each key is a client ID and each value is a tuple (x, y) 
    representing the client's data and corresponding labels.
    """
    client_data = {}
    shard_size = len(x_train) // num_shards
    
    # Sort by label to group the same classes together
    sorted_indices = np.argsort(y_train.flatten())
    x_train_sorted = x_train[sorted_indices]
    y_train_sorted = y_train[sorted_indices]

    # Each shard contains only one class
    shards = [(x_train_sorted[i * shard_size: (i + 1) * shard_size],
               y_train_sorted[i * shard_size: (i + 1) * shard_size]) for i in range(num_shards)]
    
    shards_per_client = num_shards // num_clients

    # List of class labels to ensure each client has some data from every class
    unique_classes = np.unique(y_train)

    for i in range(num_clients):
        # Select two classes at random which will be the focused ones
        main_classes = np.random.choice(unique_classes, size=2, replace=False)
        
        client_x, client_y = [], []
        
        # Add majority of the data from the two main classes
        for cls in main_classes:
            class_indices = np.where(y_train_sorted == cls)[0]
            num_samples_focus = int(focus_ratio * shard_size * shards_per_client // 2)  # Focused data
            selected_indices = np.random.choice(class_indices, size=num_samples_focus, replace=False)
            client_x.append(x_train_sorted[selected_indices])
            client_y.append(y_train_sorted[selected_indices])
        
        # Add a small portion of random data from other classes
        remaining_classes = np.setdiff1d(unique_classes, main_classes)
        num_samples_other = int((1 - focus_ratio) * shard_size * shards_per_client)
        
        # Determine how to distribute the remaining data across other classes
        class_distribution = np.random.dirichlet(np.ones(len(remaining_classes)), size=1)[0]
        class_distribution = class_distribution / class_distribution.sum()  # Normalize to ensure it sums to 1
        
        for cls, dist in zip(remaining_classes, class_distribution):
            class_indices = np.where(y_train_sorted == cls)[0]
            num_samples_per_class = int(dist * num_samples_other)
            # Ensure we don't try to select more samples than available
            num_samples_per_class = min(num_samples_per_class, len(class_indices))
            selected_indices = np.random.choice(class_indices, size=num_samples_per_class, replace=False)
            client_x.append(x_train_sorted[selected_indices])
            client_y.append(y_train_sorted[selected_indices])

        # Concatenate the data and shuffle to mix the classes
        client_x = np.vstack(client_x)
        client_y = np.vstack(client_y)
        shuffle_indices = np.random.permutation(len(client_x))
        client_x = client_x[shuffle_indices]
        client_y = client_y[shuffle_indices]

        # Assign the data to the client
        client_data[i] = (client_x, client_y)
    
    return client_data
