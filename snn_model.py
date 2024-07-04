import tensorflow as tf
from tensorflow.keras import layers, models

class SiameseNetwork:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self, hp):
        def create_base_network(input_shape):
            input = layers.Input(shape=input_shape)
            x = layers.Conv1D(64, 3, activation='relu')(input)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Conv1D(128, 3, activation='relu')(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Flatten()(x)
            x = layers.Dense(hp.Choice('fc_units', [32, 64, 128]), activation='relu')(x)
            x = layers.Dropout(hp.Choice('dropout_rate', [0.1, 0.2, 0.3]))(x)
            x = layers.Dense(hp.Choice('embedding_dim', [32, 64, 128]), activation='relu')(x)
            return models.Model(input, x)

        input_a = layers.Input(shape=self.input_shape)
        input_b = layers.Input(shape=self.input_shape)

        base_network = create_base_network(self.input_shape)

        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = layers.Lambda(lambda embeddings: tf.sqrt(tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1, keepdims=True)))([processed_a, processed_b])

        model = models.Model([input_a, input_b], distance)
        model.compile(
            loss=self.contrastive_loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [0.001, 0.0001])),
            metrics=['accuracy']
        )

        return model

    def contrastive_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(hp.Choice('margin', [0.5, 1.0, 1.5]) - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

    def train(self, model, train_dataset, val_dataset, epochs=20):
        with tf.device('/GPU:0'):
            model.fit(train_dataset,
                      validation_data=val_dataset,
                      epochs=epochs)
