import tensorflow as tf


class ModelGenerator:
    def __init__(self, time_steps, input_height, input_width, input_channels):
        self.input_shape = (time_steps, input_height, input_width, input_channels)

    def generate_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 37), strides=(4, 4), padding="same"),
                input_shape=self.input_shape),
            tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU()),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(2, 2))),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.3)),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding="same")),
            tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU()),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(2, 2))),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.3)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=512)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.LeakyReLU()),
            tf.keras.layers.LSTM(units=512, return_sequences=True, dropout=0.2),
            tf.keras.layers.LSTM(units=222, return_sequences=True, dropout=0.3),
            tf.keras.layers.Lambda(lambda x: x[:, -1, :]),
            tf.keras.layers.Dense(units=222),
            tf.keras.layers.Softmax()
        ])

        return model
