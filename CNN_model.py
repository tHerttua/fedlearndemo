import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


def create_model(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):

    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)  

    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(  optimizer=optimizer, 
                    loss=loss, 
                    metrics=metrics)

    return model