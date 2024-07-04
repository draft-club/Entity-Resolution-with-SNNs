import kerastuner as kt
from snn_model import SiameseNetwork
from hyperparams import Hyperparams

class SNNTuner:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.hyperparams = Hyperparams().get_params()
        self.tuner = kt.Hyperband(
            self.build_model,
            objective='val_accuracy',
            max_epochs=20,
            hyperband_iterations=2
        )

    def build_model(self, hp):
        snn = SiameseNetwork(self.input_shape)
        return snn.build_model(hp)

    def run_tuning(self, pairs_train, labels_train, pairs_val, labels_val):
        self.tuner.search(
            [pairs_train[:, 0], pairs_train[:, 1]], labels_train,
            validation_data=([pairs_val[:, 0], pairs_val[:, 1]], labels_val),
            epochs=20,
            batch_size=128
        )
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        return best_hps
