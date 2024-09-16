import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


def create_model(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
    """
    Convolutional layers with convolutional filters
    Pooling layers downsamples the feature maps to reduce computational cost
    Flatten layer converts the 2D map into a vector
    ReLU activation is efficient and is placed to prevent vanishing gradient.
    Dense layers process the features into 10 different classes (as per Cifar-10)
    Softmax normalizes the output probabilities to sum up to 1 between the classes

    optimizer, loss and metrics can be passed, but the preset ones are:
    Adam as the optimizer as it is popular and results in fast convergence.
    sparse_categorical_crossentropy because it is suitable for multiclass classification.
    Accuracy as the default metrics to track how often the prediction is correct.

    In this particular dataset it is not critical to monitor for example false positives
    too closely, so following accuracy is enough.

    returns the compiled model
    """

    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)  

    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(  optimizer=optimizer, 
                    loss=loss, 
                    metrics=metrics)

    return model