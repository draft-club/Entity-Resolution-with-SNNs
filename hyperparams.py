from kerastuner import HyperParameters

class Hyperparams:
    def __init__(self):
        self.hp = HyperParameters()

    def get_params(self):
        self.hp.Choice('embedding_dim', [32, 64, 128])
        self.hp.Choice('lstm_units', [64, 128, 256])
        self.hp.Choice('fc_units', [32, 64, 128])
        self.hp.Choice('dropout_rate', [0.1, 0.2, 0.3])
        self.hp.Choice('learning_rate', [0.001, 0.0001])
        self.hp.Choice('margin', [0.5, 1.0, 1.5])
        return self.hp
